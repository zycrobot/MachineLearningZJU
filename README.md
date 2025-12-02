# MachineLearningZJU


## Course Information
- **Course Name**: Machine Learning
- **Instructor**: Professor Deng Cai
- **Semester**: Fall 2020


## Project Structure


```
MachineLearningZJU/
├── README.md                 # Project documentation
├── assignment1/              # Assignment 1
│   ├── hw1(4).pdf            # Assignment description
│   ├── hw1_code(1)/          # Code implementation
│   │   └── ml2020fall_hw1/
│   │       ├── bayes_decision_rule/     # Bayes decision rule
│   │       ├── gaussian_discriminant/   # Gaussian discriminant analysis
│   │       └── text_classification/     # Text classification
│   ├── hw1_data(1).zip       # Dataset
│   └── report.pdf            # Experiment report
├── assignment2/              # Assignment 2
│   ├── hw2(2).pdf            # Assignment description
│   ├── hw2_code(1)/          # Code implementation
│   │   └── ml2020fall_hw2/
│   │       ├── linear-models/           # Linear models
│   │       └── regularization-cross-validation/ # Regularization and cross-validation
│   └── report.pdf            # Experiment report
├── assignment3/              # Assignment 3
│   ├── hw3(1).pdf            # Assignment description
│   ├── hw3/                  # Code implementation
│   │   └── ml2020fall_hw3/
│   │       └── neural_network/          # Neural network
│   └── report.pdf            # Experiment report
├── assignment4/              # Assignment 4
│   ├── 2020fall_hw4_report.pdf # Experiment report
│   ├── hw4(1).pdf            # Assignment description
│   └── hw4_code(1)/          # Code implementation
│       └── ml2020fall_hw4/
├── assignment5/              # Assignment 5
│   ├── 2020fall_hw5_report.pdf # Experiment report
│   ├── hw5.pdf               # Assignment description
│   └── hw5_code/             # Code implementation
│       └── ml2020fall_hw5/
└── assignment6/              # Assignment 6
    ├── 2020fall_hw6_report.pdf # Experiment report
    ├── hw6.pdf               # Assignment description
    └── hw6_code/             # Code implementation
        └── ml2020fall_hw6/
```

## Assignment Overview

### assignment1: Bayesian Classification Methods

This assignment mainly consists of three parts: Bayes decision rule, Gaussian discriminant analysis, and text classification.

1. **Bayes Decision Rule**
   - **Principle**: Based on Bayes' theorem, combining prior probability and likelihood probability to calculate posterior probability, then making decisions according to the minimum risk principle
   - **Implementation**:
     - `likelihood.py`: Calculate feature likelihood probability by dividing feature values by the total feature values of each class
     - `posterior.py`: Calculate posterior probability using Bayes' formula, combining likelihood probability and prior probability
     - `get_x_distribution.py`: Process training and testing data distribution
   - **Application**: Solve classification problems with discrete features, implementing minimum total risk decision

2. **Gaussian Discriminant Analysis**
   - **Principle**: Assume that features of different classes follow different Gaussian distributions, learn model parameters through maximum likelihood estimation
   - **Implementation**:
     - `gaussian_pos_prob.py`: Calculate posterior probability for Gaussian discriminant analysis
     - Use Gaussian distribution formula, matrix inversion and determinant calculation for likelihood probability
     - Calculate posterior probability through Bayes' formula
   - **Application**: Handle classification problems with continuous features, assuming data follows Gaussian distribution

3. **Text Classification**
   - **Principle**: Based on Naive Bayes, assuming conditional independence between features, calculating likelihood probability through word frequency statistics
   - **Implementation**:
     - `likelihood.py`: Calculate likelihood probabilities of features for different classes
     - Use smoothing techniques to handle unseen words
   - **Application**: Spam email detection, identifying spam and normal emails through word frequency analysis

### assignment2: Linear Models and Regularization

This assignment mainly consists of two parts: linear models and regularization with cross-validation.

1. **Linear Models**
   - **Linear Regression**
     - **Principle**: Learn linear model parameters by minimizing mean squared error
     - **Implementation**: `linear_regression.py`, add bias term and calculate weights using matrix inversion
     - **Application**: Continuous value prediction problems

   - **Logistic Regression**
     - **Principle**: Use sigmoid function to map linear output to probability space, learn parameters through maximum likelihood estimation
     - **Implementation**: `logistic.py`, contains sigmoid function and stochastic gradient descent optimization process
     - **Application**: Binary classification problems, outputting class probabilities

   - **Perceptron**
     - **Principle**: Update weights through misclassified points to achieve linear classification
     - **Implementation**: `perceptron.py`, iterative update process with bias term, setting maximum 200 iterations and convergence conditions
     - **Application**: Binary classification problems with linearly separable data

   - **Support Vector Machine (SVM)**
     - **Principle**: Maximize classification margin, find optimal hyperplane
     - **Implementation**: `svm.py`, implemented using scipy.optimize.minimize, contains linear constraint functions and objective function
     - `svm_slack.py`: Implement SVM with slack variables to handle linearly inseparable cases
     - **Application**: Linear and non-linear classification problems

2. **Regularization and Cross-validation**
   - **Ridge Regression**
     - **Principle**: Add L2 regularization term to linear regression to prevent overfitting
     - **Implementation**: `ridge.py`, add bias term and calculate weights using matrix inversion, contains regularization term
     - **Application**: Handling multicollinearity problems and preventing overfitting

   - **Regularized Logistic Regression**
     - **Principle**: Add L2 regularization term to logistic regression to balance model complexity and fitting ability
     - **Implementation**: `logistic_r.py`, contains sigmoid function, stochastic gradient descent optimization (with regularization term) and convergence condition judgment
     - **Application**: Preventing overfitting in logistic regression models
   - **Cross-validation**
     - **Implementation**: `validation.ipynb`, select optimal regularization parameters through cross-validation
     - **Application**: Model selection and hyperparameter tuning

### assignment3: Neural Networks

This assignment mainly implements neural network algorithms, including fully connected neural networks and convolutional neural networks, for image classification tasks.

1. **Fully Connected Neural Network**
   - **Principle**: Multiple layers of neurons connected in a fully connected manner, using non-linear activation functions to introduce non-linear capabilities
   - **Implementation**:
     - `fc_net.py`: Implement a two-layer fully connected neural network (TwoLayerNet)
     - Architecture: Input layer -> Fully connected -> ReLU -> Fully connected -> Softmax
     - Initialize weights with Gaussian distribution, zero-initialize biases
     - Implement forward propagation to calculate loss and backpropagation to calculate gradients
     - Include L2 regularization to prevent overfitting
   - **Application**: MNIST handwritten digit recognition and other image classification tasks

2. **Convolutional Neural Network (CNN)**
   - **Principle**: Use convolutional layers to extract local features, pooling layers to reduce dimensionality, fully connected layers for classification
   - **Implementation**:
     - `cnn.py`: Implement a three-layer convolutional neural network (ThreeLayerConvNet)
     - Architecture: Convolutional layer -> ReLU -> Max pooling -> Fully connected -> ReLU -> Fully connected -> Softmax
     - Use convolution operations to extract image features, pooling layers to reduce feature dimensionality
   - **Application**: Complex image classification tasks, extracting hierarchical features

3. **Optimizers**
   - **Implementation**: `optim.py`, contains multiple optimization algorithms
     - `sgd`: Stochastic gradient descent
     - `sgd_momentum`: Stochastic gradient descent with momentum, accelerating convergence and reducing oscillations
   - **Training Process**:
     - `solver.py`: Encapsulate training logic, including batch processing, learning rate decay, early stopping and other mechanisms
     - Support performance monitoring on training and validation sets, preventing overfitting

### assignment4: K-Nearest Neighbors and Ensemble Learning

This assignment mainly implements K-nearest neighbors (KNN) algorithm and ensemble learning algorithms, including decision trees, random forests, and AdaBoost, for classification tasks.

1. **K-Nearest Neighbors (KNN) Algorithm**
   - **Principle**: Calculate distances between test samples and training samples, select K nearest neighbors for voting decisions
   - **Implementation**:
     - `knn.py`: Implement KNN classifier
     - Use Euclidean distance to measure sample similarity
     - Sort distances using numpy.argsort, perform majority voting with scipy.stats.mode
   - **Application**: Image classification and other tasks, non-parametric method that does not require training

2. **Decision Tree**
   - **Principle**: Recursively partition feature space to build a tree-structured decision model
   - **Implementation**:
     - `decision_tree.py`: Implement decision tree classifier
     - Support multiple splitting criteria: information gain, information gain ratio, Gini index
     - Include pruning parameters: maximum depth, minimum number of samples for leaf nodes
   - **Application**: Highly interpretable classification tasks, suitable for handling mixed-type features

3. **Random Forest**
   - **Principle**: Ensemble multiple decision trees, reduce variance through Bagging (bootstrap sampling) and random feature selection
   - **Implementation**:
     - `random_forest.py`: Implement random forest classifier
     - Perform bootstrap sampling (with replacement) for each base tree
     - Each tree node uses only √p random features
     - Make ensemble predictions through majority voting
   - **Application**: High-dimensional data classification, usually has better generalization ability than single decision tree

4. **AdaBoost**
   - **Principle**: Adaptive boosting, train multiple weak classifiers using weighted samples, combine them according to weights to form a strong classifier
   - **Implementation**:
     - `adaboost.py`: Implement AdaBoost classifier
     - Update sample weights and classifier weights based on previous error rate
     - Classifier weights based on logarithmic ratio of error rates
     - Make predictions through weighted voting
   - **Application**: Improving weak classifier performance, handling binary classification problems

### assignment5: Unsupervised Learning

This assignment mainly implements unsupervised learning algorithms, including K-means clustering, Principal Component Analysis (PCA), and spectral clustering, for data clustering and dimensionality reduction tasks.

1. **K-means Clustering**
   - **Principle**: Divide data into K clusters, minimizing the sum of squared distances from samples to cluster centers within clusters
   - **Implementation**:
     - `kmeans.py`: Implement K-means clustering algorithm
     - Randomly select K initial cluster centers
     - Iterative update: assign samples to nearest cluster centers, recalculate cluster centers
     - Until cluster centers no longer change or maximum number of iterations is reached
   - **Application**: Image compression, data grouping, feature quantization and other tasks

2. **Principal Component Analysis (PCA)**
   - **Principle**: Project data onto directions of maximum variance through linear transformation to achieve dimensionality reduction
   - **Implementation**:
     - `pca.py`: Implement PCA algorithm
     - Data centering processing
     - Calculate eigenvalues and eigenvectors of covariance matrix
     - Select top k eigenvectors with largest eigenvalues as principal components
   - **Application**: Data dimensionality reduction, feature extraction, visualization and other tasks

3. **Spectral Clustering**
   - **Principle**: Graph theory-based clustering method, treating data as graph nodes, performing clustering through similarity graph construction and spectral decomposition
   - **Implementation**:
     - `spectral.py`: Implement spectral clustering algorithm
     - Build adjacency matrix W
     - Calculate degree matrix D and Laplacian matrix L = D - W
     - Perform eigenvalue decomposition on Laplacian matrix
     - Use K-means to cluster feature vectors
   - **Application**: Complex shape data clustering, capable of discovering non-convex clustering structures

### assignment6: Reinforcement Learning

This assignment mainly implements reinforcement learning algorithms, including Q-learning and neural network-based function approximation methods, for solving reinforcement learning problems.

1. **Tabular Q-learning**
   - **Principle**: Store state-action values through Q-tables, use ε-greedy strategy to balance exploration and exploitation
   - **Implementation**:
     - `qTable.py`: Implement tabular Q-learning algorithm
     - Update Q values using Bellman equation: Q(s,a) += α[ r + γ max_a' Q(s',a') - Q(s,a) ]
     - Implement ε-greedy strategy for action selection, with ε decaying over time
   - **Application**: Small state space reinforcement learning problems, such as simple control tasks

2. **Neural Network-based Function Approximation**
   - **Principle**: Use neural networks to approximate Q functions, handling large state space problems
   - **Implementation**:
     - `approximator_torch.py`: Implement neural network function approximation using PyTorch
     - Include simple linear models and convolutional neural network models (for image input)
     - Implement experience replay (Replay Memory) storage and sampling
     - Support DQN and Double DQN algorithms
   - **Application**: Complex control problems, such as Atari games and other large state space problems

3. **Training and Evaluation**
   - Implement complete reinforcement learning training loop
   - Record and visualize reward changes during training
   - Evaluate performance of trained policies

4. **Extensions**
   - Support different exploration strategies
   - Implement learning rate scheduling and hyperparameter optimization
   - Support different environment interfaces


## Environment Requirements
This course assignments are mainly implemented using Python. Recommended environment configuration:
- Python 3.6+
- NumPy
- SciPy
- Pandas
- Matplotlib
- Scikit-learn
- PyTorch

