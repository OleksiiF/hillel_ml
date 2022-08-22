##### Dataset
* prepared
* removed anomaly
* balanced (equal quantity of classes)
  * increase amount of the minor classes
    * augmentation method
        * picture
          * mirroring
          * rotate
  * reduce amount of the major classes

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
* [streamlit](https://streamlit.io)

##### Typical issues
* Curse of dimensionality - in case of large data dimension (a lot of features) classes can be far away from each other and doesn't match one class
* Feature scale - an issue of different scale for features (the model will assume that the price of a house equally depends on both the footage to the metro and the number of rooms --> delta 3 rooms ~ delta 3 meters to the metro)

##### Issues solutions
* PCA (Principal Component Analysis) - method of reducing data dimensionality and noise in data
  * Only works with numbers
  * Scale sensitive (requires data standardization before use)
  * Outliers are highly impacted (requires removal/limitation)
* Normalization of scale
  * Subtract the mean from the feature value and divide by the standard deviation
  * Each feature normalized by specified number (weight)

##### Models
* KNN (K Nearest Neighbors)
  * Type of task 
    * classification (classes)
    * regression (1,2,3,4,5)
  * Issues: curse of dimensionality, feature scale

##### Glossary
* Convergence - an iterative process of reducing the difference between the previous and current value of a function
  * Accuracy - criterion for convergence
* Distribution function - description of general/all data
* Elementary outcome - one event
* Epoch - 1 iteration trough learning dataset
* Function of the activation - function which determines the neuron's output signal
* Empirical risk - mean of loss function
* Histogram - description of the data sample
* Hyperparameter - setup before learning and depend by engineer
  * Learning rate
  * Hidden layer
  * Epoch quantity
* Learning rate - indicates which part of the information is still and which will be used for recalculation
* Model of ML, function of ML - a function, which describes the dataset, often it is distribution function
* Optimization - set of methods to find the coefficients/weights of Model/NN, which provide the extremums (usually minimums) of the error's function/target function/loss function.
  * Method of the least squares
  * Method of most likelihood
* Probability
  * Winner, Colmagor - the ratio of favorable outcomes to the number of all outcomes
  * Bayesian probability (conditional probability) -> P(A|B) = P(B|A) * P(A) / P(B)
    * P(A) - apriori probability, coef/weights before experiment
    * P(A|B) - posteriori probability (after each iteration become P(A))
* Representation of the dataset - feature of the dataset to reproduce properties of general/all data