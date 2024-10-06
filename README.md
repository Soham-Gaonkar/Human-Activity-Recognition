# Human Activity Recognition (HAR)

Human Activity Recognition (HAR) refers to the capability of machines to identify various activities performed by the users. The knowledge acquired from these systems/algorithms is integrated into many applications where the associated device uses it to identify actions or gestures and performs predefined tasks in response.

## Project Overview
This project focuses on Human Activity Recognition (HAR) using machine learning techniques, specifically decision trees. The primary dataset used is the UCI-HAR dataset, which consists of time-series data from 30 subjects performing six different activities.

### Key Highlights:
- **Data Analysis & Feature Extraction:** Conducted an in-depth data analysis of the UCI-HAR dataset.
- **Feature Extraction using TSFEL:** Utilized the TSFEL library for comprehensive feature extraction and applied Principal Component Analysis (PCA)
to reduce data dimensionality.
- **Model Training:** Trained a Decision Tree model on the featurized dataset and evaluated its performance.
- **Custom Data Collection:** Collected g-force acceleration data from a smartphone using the Physics Toolbox Suite app and tested the trained model on this custom dataset, achieving 66% accuracy and 67% precision.
- **Custom Decision Tree Implementation:** Implemented a Decision Tree algorithm from scratch, supporting both real and discrete input-output combinations. This custom implementation achieved comparable performance to Scikit-learn’s Decision Tree classifier.
- **Visualization:** Integrated Graphviz to generate clear visual representations of the decision tree structure for better interpretability.


## Dataset
We classify human activities based on accelerometer data using a publicly available dataset called [UCI-HAR](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8567275). The dataset can be downloaded [here](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones). A video of the authors collecting participant's accelerometer data is also available [here](http://www.youtube.com/watch?v=XOEN9W05_4A).

## Installation and Setup
1. Clone the repository
    ```sh
    git clone https://github.com/Soham-Gaonkar/Human-Activity-Recognition.git
    ```
2. Change into the repository directory:
    ```sh
    cd Human-Activity-Recognition
    ```
3. Set up a virtual environment:

    a) for Unix-based platforms like Linux and macOS
    ```sh
    python -m venv venv
    source venv/bin/activate
    ```

    b) for Windows
    ```sh
    python -m venv venv
    venv\Scripts\activate
    ```

4. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```



## Task 1 : Exploratory Data Analysis (EDA) [3 marks]

### Preprocessing
We use the raw accelerometer data within the inertial_signals folder. The provided script, `CombineScript.py`, organizes and sorts accelerometer data, establishing separate classes for each category and compiling participant data into these classes. `MakeDataset.py` script is used to read through all the participant data and create a single dataset. The dataset is then split into train,test and validation set. We focus on the first 10 seconds of activity, translating to the initial 500 data samples due to a sampling rate of 50Hz.

* **Step-1>** Place the `CombineScript.py` and `MakeDataset.py` in the same folder that contains the UCI dataset. Ensure you have moved into the folder before running the scripts. If you are runing the scripts from a different folder, you will have to play around with the paths in the scripts to make it work.
* **Step-2>** Run `CombineScript.py` and provide the paths to test and train folders in UCI dataset. This will create a folder called `Combined` which will contain all the data from all the participants. This is how most of the datasets are organized. You may encounter similar dataset structures in the future.
* **Step-3>** Run `MakeDataset.py` and provide the path to `Combined` folder. This will create a Dataset which will contain the train, test and validation set. You can use this dataset to train your models.


### Questions

1. Plot the waveform for one sample data from each activity class. Are you able to see any difference/similarities between the activities? You can plot a subplot having 6 columns to show differences/similarities between the activities. Do you think the model will be able to classify the activities based on the data?
2. Do you think we need a machine learning model to differentiate between static activities (laying, sitting, standing) and dynamic activities(walking, walking_downstairs, walking_upstairs)? Look at the linear acceleration $(acc_x^2+acc_y^2+acc_z^2)$ for each activity and justify your answer.
3. Visualize the data using PCA.
    * Use PCA (Principal Component Analysis) on Total Acceleration $(acc_x^2+acc_y^2+acc_z^2)$ to compress the acceleration timeseries into two features and plot a scatter plot to visualize different class of activities. 
    *  Next, use [TSFEL](https://tsfel.readthedocs.io/en/latest/) ([a featurizer library](https://github.com/fraunhoferportugal/tsfel)) to create features (your choice which ones you feel are useful) and then perform PCA to obtain two features. Plot a scatter plot to visualize different class of activities. 
    *  Now use the features provided by the dataset and perform PCA to obtain two features. Plot a scatter plot to visualize different class of activities.
    *  Compare the results of PCA on Total Acceleration, TSFEL and the dataset features. Which method do you think is better for visualizing the data? 
4. Calculate the correlation matrix of the features obtained by TSFEL and provided in the dataset. Identify the features that are highly correlated with each other. Are there any redundant features?


## Task 2 : Decision Trees for Human Activity Recognition [3 marks]

### Questions

1. Use Sklearn Library to train Decision Tress.
    * Train a decision tree model using the raw accelerometer data. Report the accuracy, precision, recall and confusion matrix of the model. 
    * Train a decision tree model using the features obtained by TSFEL. Report the accuracy, precision, recall and confusion matrix of the model. 
    * Train a decision tree model using the features provided in the dataset. Report the accuracy, precision, recall and confusion matrix of the model. 
    * Compare the results of the three models. Which model do you think is better? 
2. Train Decision Tree with varying depths (2-8) using all above 3 methods. Plot the accuracy of the model on test data vs the depth of the tree.
3. Are there any participants/ activitivies where the Model performace is bad? If Yes, Why?

## Task 3 : Prompt Engineering for Large Language Models (LLMs) [4 marks]

### Zero-shot and Few Shot Prompting :
Zero-shot prompting involves providing a language model with a prompt or a set of instructions that allows it to generate text or perform a task without any explicit training data or labeled examples. The model is expected to generate high-quality text or perform the task accurately based solely on the prompt and its internal knowledge.

Few-shot prompting is similar to zero-shot prompting, but it involves providing the model with a limited number of labeled examples or prompts that are relevant to the specific task or dataset. The model is then expected to generate high-quality text or perform the task accurately based on the few labeled examples and its internal knowledge.

You have been provided with a [Python notebook](./HAR/ZeroShot_FewShot.ipynb) that demonstrates how to use zero-shot and few-shot prompting with a language model (LLM). The example in the notebook involves text-based tasks, but LLMs can also be applied to a wide range of tasks (Students intrested in learning more can read [here](https://deepai.org/publication/large-language-models-are-few-shot-health-learners) and [here](https://arxiv.org/pdf/2305.15525v1)). 

#### Setup Instructions:
1. To obtain API key go to the GroqCloud Developer Console at https://console.groq.com/login. Follow the Quickstart guide to obtain your API key.
2. After obtaining the API key, create a .env file in the root directory of your project.
3. In the `.env` file, add the following line:
    ```env
    GROQ_API_KEYS=apikey1,apikey2
    ```
4. Replace apikey1, apikey2 with the actual keys you obtained from GroqCloud. You can list multiple keys separated by commas if needed.

**WARNING: Exceeding the token limit for a particular Groq model multiple times may result in a ban on your account. Please ensure that you stay within the model’s token usage limits to avoid account suspension.**

Queries are provided in the form of featurized accelerometer data and the model should predict the activity performed.
* **Zero shot learning** : The model should be able to predict the activity based on the accelerometer data without any explicit training data or labeled examples. 
* **Few Shot Learning** :The model should also be able to predict the activity based on a limited number of labeled examples or prompts that are relevant to the specific task. 

### Questions

1. Demonstrate how to use Zero-Shot Learning and Few-Shot Learning to classify human activities based on the featurized accelerometer data. Qualitatively demonstrate the performance of Few-Shot Learning with Zero-Shot Learning. Which method performs better? Why?
2. Quantitatively compare the accuracy of Few-Shot Learning with Decision Trees (You may use a subset of the test set if you encounter rate-limiting issues). Which method performs better? Why?
3. What are the limitations of Zero-Shot Learning and Few-Shot Learning in the context of classifying human activities based on featurized accelerometer data?
4. What does the model classify when given input from an entirely new activity that it hasn't seen before? 
5. Test the model with random data (ensuring the data has the same dimensions and range as the previous input) and report the results. 


## Task 4 : Data Collection in the Wild [4 marks]

## Task Description
Utilize apps like `Physics Toolbox Suite` from your smartphone to collect your data in .csv/.txt format. Ensure at least 15 seconds of data is collected, trimming edges to obtain 10 seconds of relevant data. Collect 3-5 samples per activity class.

### Things to take care of:
* Ensure the phone is placed in the same position for all the activities.
* Ensure the phone is in the same alignment during the activity as changing the alignment will change the data collected and will affect the model's performance.
* Ensure to have atleast 10s of data per file for training. As the data is collected at 50Hz, you will have 500 data samples.

### Questions
1. Use the Decision Tree model trained on the UCI-HAR dataset to predict the activities that you performed. Report the accuracy, precision, recall and confusion matrix of the model. You have three version of UCI dataset you can use a)Raw data from accelerometer, b)TSFEL featurised data, c)Features provided by author. Choose which version to use, ensuring that your test data is similar to your training data. How did the model perform? 
2. Use the data you collected to predict the activities that you performed. Decide whether to apply preprocessing and featurization, and if so, choose the appropriate methods. How did the model perform? 
3. Use the Few-Shot prompting method using UCI-HAR dataset to predict the activities that you performed. Ensure that both your examples and test query undergo similar preprocessing. How did the model perform? 
4. Use the Few-Shot prompting method using the data you collected to predict the activities that you performed. Adopt proper processing methods as needed. How did the model perform? 


## Decision Tree Implementation [6 marks]

1. Complete the decision tree implementation in tree/base.py. 

    We have implemented the decision tree in tree/base.py in Python, without using any external libraries except those shared in class or already imported in the codebase. The decision tree successfully handles all four cases:

    -  Discrete features, discrete output
    - Discrete features, real output
    - Real features, discrete output
    - Real features, real output

    The decision tree uses Information Gain with Entropy or Gini Index as the splitting criteria for discrete output, and Information Gain with MSE for real output.

    We have used Graphviz to visualize and plot the decision tree graphically. Additionally, the tree is printed in the terminal with appropriate formatting. Both include feature names and split criteria, for a clearer understanding of the structure.
    
- `metrics.py`: Performance metrics functions.

- `usage.py`: Run this file to check the implemented decision tree.

- tree (Directory): Module for decision tree.
    - `base.py` : Decision Node and Decision Tree Class and Graphviz plotting functions.
    - `utils.py`: Utility functions.
    - `__init__.py`: **Do not edit this**


2. Generate your dataset using the following lines of code

    ```python
    from sklearn.datasets import make_classification
    X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

    # For plotting
    import matplotlib.pyplot as plt
    plt.scatter(X[:, 0], X[:, 1], c=y)
    ```

    a) Show the usage of *your decision tree* on the above dataset. The first 70% of the data should be used for training purposes and the remaining 30% for test purposes. Show the accuracy, per-class precision and recall of the decision tree you implemented on the test dataset. 

    b) Use 5 fold cross-validation on the dataset. Using nested cross-validation find the optimum depth of the tree. 
    
    > `classification-exp.py` contains the code for the above experiments. It is also available in .ipynb format in the Task-5 folder for better clarity and ease of execution. 


3. 
    a) Show the usage of your decision tree for the [automotive efficiency](https://archive.ics.uci.edu/ml/datasets/auto+mpg) problem. 

    b) Compare the performance of your model with the decision tree module from scikit learn. 
    
   > `auto-efficiency.py` contains the code for the above experiments. It is also available in .ipynb format in the Task-5 folder for better clarity and ease of execution.
    
4. Create some fake data to do some experiments on the runtime complexity of your decision tree algorithm. Create a dataset with N samples and M binary features. Vary M and N to plot the time taken for: 1) learning the tree, 2) predicting for test data. How do these results compare with theoretical time complexity for decision tree creation and prediction. You should do the comparison for all the four cases of decision trees. 

    >`experiments.py` contains the code for the above experiments. It is also available in .ipynb format in the Task-5 folder for better clarity and ease of execution.


The answers to all the subjective questions (visualization, timing analysis, plots) are also available in the `Answers` folder present in the root directory.


## Acknowledgements
We would like to extend our gratitude to [Prof. Nipun Batra](https://nipunbatra.github.io/) for his guidance, support, and for providing us with the opportunity to work on this project as part of the ES 335: Machine Learning course at IIT Gandhinagar.


## License
This project is licensed under the [MIT License](https://github.com/ShardulJunagade/ES-335-Assignment-1-2024-Fall/blob/main/LICENSE) - see the [LICENSE](https://github.com/ShardulJunagade/ES-335-Assignment-1-2024-Fall/blob/main/LICENSE) file for details.