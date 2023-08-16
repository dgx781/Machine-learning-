# Machine Learning
## A Brief Introduction:-
Machine learning (ML) is a subdomain of artificial intelligence (AI) that focuses on developing systems that learn—or improve performance—based on the data they ingest. Artificial intelligence is a broad word that refers to systems or machines that resemble human intelligence. Machine learning and AI are frequently discussed together, and the terms are occasionally used interchangeably, although they do not signify the same thing. A crucial distinction is that, while all machine learning is AI, not all AI is machine learning.

Machine learning (ML) is an umbrella term for solving problems for which development of algorithms by human programmers would be cost-prohibitive, and instead the problems are solved by helping machines 'discover' their 'own' algorithms, without needing to be explicitly told what to do by any human-developed algorithms. ML learns from the data it is given as input. Machines only understand Binary Language in the form of 0's and 1's, the data which is collected from various sources which are private or public are transformed from raw data to clean machine readable format, then all the categorical variables are converted into numerical variables and that preprocessed data is fed to the ML model which it learns by using various suitable lgorithms depending on the type of problem statement. 

For more info you can refer to:- https://www.geeksforgeeks.org/general-steps-to-follow-in-a-machine-learning-problem/

## Various Steps in Machine Learning:-
### Finding the Objective of the problem and Formulating Related Questions:-
What is it that we want to find out? How will we reach the success criteria that we set?

Let’s say we are performing machine learning for a high-traffic fast-casual restaurant chain, and our goal is to improve the customer experience. We can serve this goal in many ways. When we’re thinking about creating a model, we have to narrow down to one measurable, specific task. For example, we might say we want to predict the wait times for customers’ food orders within 2 minutes, so that we can give them an accurate time estimate.

### Finding and Understanding the Data:-
Arguably the largest chunk of time in any machine learning process is finding the relevant data to help answer your question, and getting it into the format necessary for performing predictive analysis.

We know that for supervised learning, we need labeled datasets, or datasets that have clear labels of what their ground truth is. For an example like the restaurant wait time, this would mean we would need many examples of past orders, tagged with how long the wait time was. Maybe the restaurant already tracks this data, but we might need to augment the data collection with a timer that starts when the customer orders, stops when the customer receives their food, and records that information.

### Data Collection and Integration:-
1. The first step of the ML pipeline involves the collection of data and integration of data.
2. Data collected acts as an input to the model (data preparation phase)
3. Inputs are called features.
4. Data collected in the case of our considered example involves a lot of data. The collected data should answer the following questions- What is past customer history? What were the past orders? Is the customer a prime member of our bookstore? Does the customer own a kindle? Has the customer made any previous complaints? What was the most number of complaints?
5. The more the data is, more the better our model becomes.
6. Once the data is collected we need to integrate and prepare the data.
7. Integration of data means placing all related data together.
8. Then data preparation phase starts in which we manually and critically explore the data.
9. The data preparation phase tells the developer that is the data matching the expectations. Is there enough info to make an accurate prediction? Is the data consistent?

### Exploratory Data Analysis and Visualization:-
1. Once the data is prepared developer needs to visualize the data to have a better understanding of relationships within the dataset.
2. When we get to see data, we can notice the unseen patterns that we may not have noticed in the first phase.
3. It helps developers easily identify missing data and outliers.
4. Data visualization can be done by plotting histograms, scatter plots, etc.
5. After visualization is done data is analyzed so that developer can decide what ML technique he may use.
6. In the considered example case unsupervised learning may be used to analyze customer purchasing habits.

### Feature Selection and Engineering:-
1. Feature selection means selecting what features the developer wants to use within the model.
2. Features should be selected so that a minimum correlation exists between them and a maximum correlation exists between the selected features and output.
3. Feature engineering is the process to manipulate the original data into new and potential data that has a lot many features within it.
4. In simple words Feature engineering is converting raw data into useful data or getting the maximum out of the original data.
5. Feature engineering is arguably the most crucial and time-consuming step of the ML pipeline.
6. Feature selection and engineering answers questions – Are these features going to make any sense in our prediction?
7. It deals with the accuracy and precision of data.

### Model Training:-
1. After the first three steps are done completely we enter the model training phase.
2. It is the first step officially when the developer gets to train the model on basis of data.
3. To train the model, data is split into three parts- Training data, validation data, and test data.
4. Around 70%-80% of data goes into the training data set which is used in training the model.
5. Validation data is also known as development set or dev set and is used to avoid overfitting or underfitting situations i.e. enabling hyperparameter tuning.
6. Hyperparameter tuning is a technique used to combat overfitting and underfitting.
7. Validation data is used during model evaluation.
8. Around 10%-15% of data is used as validation data.
9. Rest 10%-15% of data goes into the test data set. Test data set is used for testing after the model preparation.
10. It is crucial to randomize data sets while splitting the data to get an accurate model.
11. Data can be randomized using Scikit learn in python.

### Model Evaluation:-
1. After the model training, validation, or development data is used to evaluate the model.
2. To get the most accurate predictions to test data may be used for further model evaluation.
3. A confusion matrix is created after model evaluation to calculate accuracy and precision numerically.
4. After model evaluation, our model enters the final stage that is prediction.

### Model Prediction:-
1. In the prediction phase developer deploys the model.
2. After model deployment, it becomes ready to make predictions.
3. Predictions are made on training data and test data to have a better understanding of the build model.

## Types of Machine Learning:-

### Supervised Machine Learning:-
As its name suggests, Supervised machine learning is based on supervision. It means in the supervised learning technique, we train the machines using the "labelled" dataset, and based on the training, the machine predicts the output. Here, the labelled data specifies that some of the inputs are already mapped to the output. More preciously, we can say; first, we train the machine with the input and corresponding output, and then we ask the machine to predict the output using the test dataset.The main goal of the supervised learning technique is to map the input variable(x) with the output variable(y). Some real-world applications of supervised learning are Risk Assessment, Fraud Detection, Spam filtering, etc.

#### Categories of Supervised Machine Learning:-
Supervised machine learning can be classified into two types of problems, which are given below:
1. Classification
2. Regression

##### Classification:-
Classification algorithms are used to solve the classification problems in which the output variable is categorical, such as "Yes" or No, Male or Female, Red or Blue, etc. The classification algorithms predict the categories present in the dataset. Some real-world examples of classification algorithms are Spam Detection, Email filtering, etc.
Some popular classification algorithms are given below:
1. Random Forest Algorithm
2. Decision Tree Algorithm
3. Logistic Regression Algorithm
4. Support Vector Machine Algorithm

##### Regression:-
Regression algorithms are used to solve regression problems in which there is a linear relationship between input and output variables. These are used to predict continuous output variables, such as market trends, weather prediction, etc.
Some popular Regression algorithms are given below:
1. Simple Linear Regression Algorithm
2. Decision Tree regression
3. Ridge Regression
4. Lasso Regression

### Unsupervised Machine Learning:-
Unsupervised Machine Learning is different from the Supervised learning technique; as its name suggests, there is no need for supervision. It means, in unsupervised machine learning, the machine is trained using the unlabeled dataset, and the machine predicts the output without any supervision. In unsupervised learning, the models are trained with the data that is neither classified nor labelled, and the model acts on that data without any supervision.

The main aim of the unsupervised learning algorithm is to group or categories the unsorted dataset according to the similarities, patterns, and differences. Machines are instructed to find the hidden patterns from the input dataset.

#### Categories of Supervised Machine Learning:-
Unsupervised Learning can be further classified into two types, which are given below:
1. Clustering
2. Association

##### Clustering:-
The clustering technique is used when we want to find the inherent groups from the data. It is a way to group the objects into a cluster such that the objects with the most similarities remain in one group and have fewer or no similarities with the objects of other groups. An example of the clustering algorithm is grouping the customers by their purchasing behaviour.
Some of the popular clustering algorithms are given below:
1. C-means Clustering
2. K-means Clustering
3. DBSCAN

##### Association:-
Association rule learning is an unsupervised learning technique, which finds interesting relations among variables within a large dataset. The main aim of this learning algorithm is to find the dependency of one data item on another data item and map those variables accordingly so that it can generate maximum profit. This algorithm is mainly applied in Market Basket analysis, Web usage mining, continuous production, etc.
Some popular algorithms of Association rule learning are:-
1.  Apriori Algorithm
2.  Eclat
3.  FP-growth algorithm.

### Reinforcement Learning:-
Reinforcement learning works on a feedback-based process, in which an AI agent (A software component) automatically explore its surrounding by hitting & trail, taking action, learning from experiences, and improving its performance. Agent gets rewarded for each good action and get punished for each bad action; hence the goal of reinforcement learning agent is to maximize the rewards.In reinforcement learning, there is no labelled data like supervised learning, and agents learn from their experiences only.

A reinforcement learning problem can be formalized using Markov Decision Process(MDP). In MDP, the agent constantly interacts with the environment and performs actions; at each action, the environment responds and generates a new state.

#### Categories of Reinforcement Learning:-
Reinforcement learning is categorized mainly into two types of methods/algorithms:-
1. Positive Reinforcement Learning
2. Negative Reinforcement Learning

##### Positive Reinforcement Learning:-
Positive reinforcement learning specifies increasing the tendency that the required behaviour would occur again by adding something. It enhances the strength of the behaviour of the agent and positively impacts it.

##### Negative Reinforcement Learning:-
Negative reinforcement learning works exactly opposite to the positive RL. It increases the tendency that the specific behaviour would occur again by avoiding the negative condition.
