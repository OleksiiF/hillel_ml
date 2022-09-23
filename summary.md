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

##### Resources
* Data sources
  * [Paperswithcode](https://paperswithcode.com)
  * [U.S. Government’s open data](https://data.gov)
  * [Datasetsearch](https://datasetsearch.research.google.com)
  * [Datahub](https://datahub.io/collections)
  * [FBI crime data](https://crime-data-explorer.fr.cloud.gov/pages/home)
* Ready to use models
  * [tfhub](https://tfhub.dev)
  * [ONNX](https://onnx.ai)
* Theory
  * [Intro](https://www.youtube.com/playlist?list=PL0Ks75aof3Thg7mpzaPFSXEUVP9PWCTAO)
  * [Dive into Deep Learning](https://d2l.ai/index.html)
  * [Geogebra](https://www.geogebra.org)
  * [DL 3Brown1Blue](https://www.youtube.com/playlist?list=PLZjXXN70PH5itkSPe6LTS-yPyl5soOovc)
  * [Singular value decomposition](https://www.youtube.com/watch?v=mfn_2d_lLxM)
* Tools
  * [Streamlit](https://streamlit.io)
  * [MLOPS](https://www.mymlops.com)
  * [Optuna](https://optuna.org)
  * [Lazypredict](https://lazypredict.readthedocs.io/en/latest)

##### Typical issues
* Curse of dimensionality - in case of large data dimension (a lot of features) classes can be far away from each other and doesn't match one class
* Feature scale - an issue of different scale for features (the model will assume that the price of a house equally depends on both the footage to the metro and the number of rooms --> delta 3 rooms ~ delta 3 meters to the metro)

##### Issues solutions
* Dimensionality reduction
  * PCA (Principal Component Analysis) - method of reducing data dimensionality and noise in data, data encrypting
    * Only works with numbers
    * Scale sensitive (requires data standardization before use)
    * Outliers are highly impacted (requires removal/limitation)
  * UMAP (Uniform Manifold Approximation and Projection)
* Normalization of scale
  * Subtract the mean from the feature value and divide by the standard deviation
  * Each feature normalized by specified number (weight)

##### Models
* KNN (K Nearest Neighbors)
  * Type of task 
    * classification (classes)
    * regression (1,2,3,4,5)
  * Issues: curse of dimensionality, feature scale

##### NN
* How much nodes in hidden layer
  * One output node, and the required input-output relationship is simple,  hidden layer dimension equal to two-thirds of the input dimension.
  * Multiple output nodes or the required input-output relationship is complex, the dimension of the hidden layer is equal to the sum of the input dimension plus the output dimension (which must be less than doubled the input dimension).
  * The input-output connection is extremely complex, the dimension of the hidden layer is one less than doubled the input dimension. 
* Perceptron
* Autoencoder
  * Task - reproduce data
  * Usefull sid effect - reduce data dimensionality

##### Tasks
* Classification
  * Target function - categorical cross-entropy  
* Prognosis
  * Target function - mean square error function

##### Derivative of the functions (of the activation, error function) 
* Hardest step of calculation for PC
* The easier way to calculate a derivative on PC (for all frameworks) is based on 
  * Replacing numerical differentiation with an analytical one
  * Decomposition with help of the Polish notation method and the corresponding calculation graph (leaves - operands, nodes - operations) allows to make the parallel calculation

##### Model training
* Signs that the model is training correctly
  * The loss function decreasing
  * Accuracy on the validation set is growing
  * Accuracy on the test set is growing
* Signs that the model is not training correctly
  * Accuracy increasing but loss function is not decreasing
  * Accuracy on the validation set is growing, but decreasing on the test set

##### Glossary
* Convergence - an iterative process of reducing the difference between the previous and current value of a function
  * Accuracy - criterion for convergence
* Distribution function - description of general/all data
* Elementary outcome - one event
* Embeddings - hidden representations of data, in view of weights
  * Autocoder better in NLP, sounds, semantic search, docs. Requires learning
  * UMAP - special case of autocoder. In case defined size of dataset, but with unknown classes. (fraud detection, pictures)
  * PCA - quick option to data overview
* Epoch - 1 iteration trough learning dataset
* Function of the activation - function which determines the neuron's output signal. Derivate is required for a spreading an error between nodes
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
* Regularization - any modification of the model to reduce its generalization error without reducing the learning error
  * Dropout
  * L1 - sum of absolute value of weights
  * L2 - sum of square of weights
* Representation of the dataset - feature of the dataset to reproduce properties of general/all data
* Validation of the model - checking on the part of data, which participated in learning
* Verification of the model - checking on the test (separate) data