(ns nnet.core
  (:gen-class)
  (:use [clojure.core.matrix :as m])
  (:import [java.util Random]))


;;;;;;;;;;;;;;;
;;;  Utils  ;;;
;;;;;;;;;;;;;;;

(defn mean
  [coll] (if (= 0 (count coll))
                         nil
                         (/ (apply + coll) (count coll))))

(defn power [base exp] (apply * (repeat base exp)))

(defn rand-norm 
  ([] (rand-norm 1 0.0 1.0))
  ([mu sigma] (rand-norm 1 mu sigma))
  ([n mu sigma]
   (repeatedly n #(+ mu (* sigma (.nextGaussian (Random.)))))))

(defn init-weights [l1 l2]
  (repeatedly l2 #(rand-norm (inc l1) 0 (/ 2 (+ (inc l1) l2)))))


;;;;;;;;;;;;;;;;;;;;;;;;
;;;  Loss Functions  ;;;
;;;;;;;;;;;;;;;;;;;;;;;;

(def mse
  {:ind_fn (fn [obs tru] (m/pow (m/sub tru obs) 2))
   :batch_fn (fn [partials] (mean partials))
   :dfn (fn [obs tru] (* -2 (m/sub tru obs)))
   })


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;  Activation Functions ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(def sigmoid
  {:fn #(/ 1 (+ 1 (exp (* -1 %))))
   :dfn #(* (/ 1 (+ 1 (exp (* -1 %)))) (- 1 (/ 1 (+ 1 (exp (* -1 %))))))})



;;;;;;;;;;;;;;;;;
;;;  Modules  ;;;
;;;;;;;;;;;;;;;;;

(defn dense-module
  [module params]
  (loop [layers (assoc (module :dense) 0 {:n (-> module :dense first)})
         i 1]
    (if (= i (count layers))
      {:dense layers}
      (recur (assoc layers i {:n (nth layers i)
                              :weights (init-weights ((nth layers (dec i)) :n)
                                                     (nth layers i))
                              :act (if (= (inc i) (count layers))
                                        (params :output_act)
                                        (params :hidden_act))})
             (inc i)))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;  Primary Functions  ;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn nnet
  "Takes a map and returns a neural network"
  ([module] (nnet module {:hidden_act 'sigmoid :output_act 'sigmoid}))
  ([module params]
   (if (= (type module) (type []))
     (nnet {:dense module} params)
     (let [params (assoc params
                         :hidden_act (if (contains? module :hidden_act)
                                       (module :hidden_act)
                                       (params :hidden_act))
                         :output_act (if (contains? module :output_act)
                                       (module :output_act)
                                       (params :output_act)))]
       (cond
         (contains? module :sequential)
         (loop [mod (module :sequential)
                i 0]
           (if (= i (count mod))
             {:sequential mod}
             (recur (assoc mod i (nnet (nth mod i) params)) (inc i))))
         (contains? module :parallel)
         (loop [mod (module :parallel)
                i 0]
           (if (= i (count mod))
             {:parallel mod
              :input_type (if (contains? module :input_type)
                            (module :input_type) :unique)}
             (recur (assoc mod i (nnet (nth mod i) params)) (inc i))))
         (contains? module :dense)
         (dense-module module params)
         )))))


(defn predict
  [model inps]
  
  (defn pred
    [model inp]
    (cond
      (contains? model :sequential)
      (loop [inp inp
             modules (model :sequential)]
        (if (= (count modules) 0)
          inp
          (recur (pred (first modules) inp) (rest modules))))
      (contains? model :parallel)
      (if (= (model :inputs) :same)
        (map #(pred % inp) (model :parallel))
        (map pred model inp))
      (contains? model :dense)
      (reduce #(map ((eval (%2 :act)) :fn)
                    (mmul (concat [1] %1) (transpose (%2 :weights))))
              inp (rest (model :dense)))))
  
  (map (partial pred model) inps))


(defn fit
  "Fits a neural network to training inputs and outputs. Must be list of inputs."
  ([model inp out] (fit model inp out
                        {:batch_size 32 :epochs 1 :lr 0.01 :loss 'mse}))
  ([model inp out params]

   (defn get-state
     [model inp]
     (cond
       (contains? model :dense)
       (loop [state [{:z inp}]
              i 1]
         (if (= (count (model :dense)) i)
           state
           (let [layer (nth (model :dense) i)
                 a (m/mmul (concat [1] ((last state) :z))
                           (m/transpose (layer :weights)))]
             (recur (conj state {:a a :z (map ((layer :act) :fn) a)})
                    (inc i)))))
       (contains? model :sequential)
       (loop [state []
              inp inp
              i 0]
         (if (= (count (model :sequential)) i)
           state
           (let [state (conj state (get-state (nth (model :sequential) i) inp))]
             (recur state ((last (last state)) :z) (inc i)))))
       (contains? model :parallel)
       (cond
         (= (model :input_type) :shared)
         (map #(get-state % inp) (model :parallel))
         (= (model :input_type) :unique)
         (map get-state (model :parallel) inp))
       ))


   (defn get-pred
     [model state]
     (cond
       (contains? model :parallel)
       (map get-pred (model :parallel) state)
       (contains? model :sequential)
       (get-pred (last (model :sequential)) (last state))
       (contains? model :dense)
       (-> state last :z)))


   (defn get-gradients
     [model state delta]
     (cond
       (contains? model :parallel)
       (loop [grads []
              deltas []
              i 0]
         (if (= i (count (model :parallel)))
           [grads deltas]
           (let [[grad delta] (get-gradients (nth (model :parallel) i)
                                             (nth state i) (nth delta i))]
             (recur (conj grads grad) (conj deltas []) (inc i)))))
       (contains? model :sequential)
       (loop [grads []
              delta delta
              i (dec (count (model :sequential)))]
         (if (= i -1)
           [grads delta]
           (let [[grad delta] (get-gradients (nth (model :sequential) i)
                                             (nth state i) delta)]
             (recur (concat [grad] grads) delta (dec i)))))
       (contains? model :dense)
       (loop [grads []
              delta delta
              i (dec (count (model :dense)))]
         (if (= i 0)
           [grads delta]
           (let [layer (nth (model :dense) i)
                 z (concat [1] ((nth state (dec i)) :z))
                 dyda (map ((layer :act) :dfn) ((nth state i) :a))
                 dydw (map #(m/mul % z) dyda)
                 grad (map m/mul delta dydw)
                 delta (map (partial apply +)
                            (transpose (map m/mul dyda (layer :weights))))]
             
             (recur (concat [grad] grads) (rest delta) (dec i)))))
       ))

   
   (defn combine-gradients
     [grad1 grad2]
     (try (m/add grad1 grad2)
          (catch Exception e (map combine-gradients grad1 grad2))))


   (defn update-weights
     [model grads params]
     (cond
       (contains? model :parallel)
       {:parallel (map #(update-weights %1 %2 params) (model :sequential) grads)}
       (contains? model :sequential)
       {:sequential (map #(update-weights %1 %2 params) (model :sequential) grads)}
       (contains? model :dense)
       (loop [model model
              i 1]
         (if (= i (count (model :dense)))
           model
           (let [weights (m/sub ((nth (model :dense) i) :weights)
                                (m/mul (params :lr) (nth grads (dec i))))]
             (recur (assoc-in model [:dense i :weights] weights) (inc i)))))))

   
   (loop [model model
          e 0]
     (if (= e (params :epochs))
       model
       (recur
        (loop [i 0
               b 1
               grads nil
               model model]
          (let [state (get-state model (nth inp i))
                delta (map (-> params :loss eval :dfn)
                           (get-pred model state) (nth out i))
                grad (first (get-gradients model state delta))
                grads (if grads (combine-gradients grads grad) grad)]
            (if (= (inc i) (count inp))
              (update-weights model grads params)
              (if (= b (params :batch_size))
                (recur (inc i) 1 nil (update-weights model grads params))
                (recur (inc i) (inc b) grads model)))))
        (inc e))))
   ))


;;;;;;;;;;;;;;
;;   Test  ;;;
;;;;;;;;;;;;;

(def inp [[[1 0] [1 0 0]]])
(def out [[[0.9 0.1 0.5] [0.9 0.1 0.5]]])
(def model (nnet {:parallel [[2 3] [2 3]]}))

(def params {:batch_size 1 :epochs 100 :lr 0.01 :loss 'mse})
(def model (time (fit model inp out params)))
(map (partial predict model) inp)

(println model)

