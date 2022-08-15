##### Dataset
* prepared
* removed anomaly
* balanced (equal quantity of classes)
  * adding of minor classes
    * augmentation method
        * picture
          * mirroring
          * rotate
  * reduce of major classes

##### Algorithm of ML for business
1. Define class of the task
2. Define method to solve the task
3. Define ground truth dataset, compare with the business dataset to find out the lack or sufficiency of features
4. Solve the task
   1. Find the ML function
   2. Optimization, reduce error of ML function

##### resources 
* [paperswithcode](https://paperswithcode.com)
  * datasets
  * methods
* tfhub
  * ready to use models
* ONNX
  * ready to use models
* https://www.youtube.com/watch?v=Ilg3gGewQ5U&ab_channel=3Blue1Brown

##### Glossary
* Elementary outcome - one event
* Distribution function - description of general/all data
* Convergence - an iterative process of reducing the difference between the previous and current value of a function
  * Accuracy - criterion for convergence
* Histogram - description of the data sample
* Model of ML, function of ML - a function, which describes the dataset, often it is distribution function
* Optimization - set of methods to find the coefficients/weights of Model/NN, which provide the extremums (usually minimums) of the error's function/target function/loss function.
  * Method of the least squares
  * Method of most likelihood
* Probability
  * Winner, Colmagor - the ratio of favorable outcomes to the number of all outcomes
  * Bayesian probability (conditional probability) -> P(A|B) = P(B|A) * P(A) / P(B)
    * P(A) - apriori probability, coef/weights before experiment
    * P(A|B) - posteriori probability (after each iteration become P(A))