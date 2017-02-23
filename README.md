# nnet

A flexible neural network library for Clojure


## Installation

    $ mvn install


## Usage

    (ns your.project
      (:use [sbhaaf/nnet.core :as nn]))


## Examples

Basic MLP:

    (nn/nnet {:dense [10 100 1]
              :hidden_act 'sigmoid  
	      :output_act 'sigmoid})
	      
or simply:

    (nn/nnet [10 100 1]) 
    

Sequential model:

    (nn/nnet {:sequential [model1 model2]} params)


Parallel model:

    (nn/nnet {:parallel [model1 model2]
              :inputs :shared})
    (nn/nnet {:parallel [model1 model2]
              :inputs :unique}) ;; default if not specified


Usage:

    (nn/predict model inps)


Training:

    (def params {:epochs 10 :batch_size 64 :lr 0.01 :loss 'mse}
    (def model (nn/fit model inps outs [params]))
    

Where loss functions can be defined:

    (def myLoss {:fn (fn [obs tru] (* (- (tru obs)) (- (tru obs))))
                 :dfn (fn [obs tru] (* -2 (- (tru obs))))})


And activation layers:

    (def myAct {:fn #(/ 1 (+ 1 (exp (* -1 %))))
                :dfn #(* (/ 1 (+ 1 (exp (* -1 %)))) (- 1 (/ 1 (+ 1 (exp (* -1 %))))))})


### TODO

* Parallel module needs work
* Need to define more loss functions and activation functions
* Rewrite structure to add layers without editing core functions
* Make single input callable from different locations
  * similar to Keras functional API
  * maybe reference by names
* Function to print model structure
* GPU!
* Efficiency

## License

MIT License

Copyright © 2017 Samuel Haaf

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.