"""
Assignment 2: regression
Goals: introduction to pandas, sklearn, linear and logistic regression, multi-class classification.
Start early, as you will spend time searching for the proper syntax, especially when using pandas
"""

import pandas
from sklearn import linear_model
import matplotlib.pyplot as plt

"""
PART 1: basic linear regression
The goal is to predict the profit of a restaurant, based on the number of habitants where the restaurant 
is located. The chain already has several restaurants in different cities. Your goal is to model 
the relationship between the profit and the populations from the cities where they are located.
Hint: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html 
"""

# Open the csv file RegressionData.csv in Excel, notepad++ or any other applications to have a 
# rough overview of the data at hand. 
# You will notice that there are several instances (rows), of 2 features (columns). 
# The values to be predicted are reported in the 2nd column.

# Load the data from the file RegressionData.csv in a pandas dataframe. Make sure all the instances 
# are imported properly. Name the first feature 'X' and the second feature 'y' (these are the labels)
data = pandas.read_csv(r"C:\Users\sande\Desktop\final semester\machine learning\assignments\A2\RegressionData.csv", header = None, names=['X', 'y']) # 5 points
# Reshape the data so that it can be processed properly
X = data['X'].values.reshape(-1,1) # 5 points
y = data['y'].values.reshape(-1,1) # 5 points
# Plot the data using a scatter plot to visualize the data
plt.scatter(X, y) # 5 points
plt.show()
# Linear regression using least squares optimization
reg = linear_model.LinearRegression() # 5 points
reg=reg.fit(X, y) # 5 points

# Plot the linear fit
fig = plt.figure()
y_pred = reg.predict(X) # 5 points
plt.scatter(X,y, c='b') # 5 points
plt.scatter(X, y_pred, c='r') # 5 points
fig.canvas.draw()

# # Complete the following print statement (replace the blanks _____ by using a command, do not hard-code the values):
print("The linear relationship between X and y was modeled according to the equation: y = b_0 + X*b_1, \
where the bias parameter b_0 is equal to ", reg.intercept_, " and the weight b_1 is equal to ", reg.coef_)
# 8 points

# Predict the profit of a restaurant, if this restaurant is located in a city of 18 habitants 
print("the profit/loss in a city with 18 habitants is ", reg.predict([[18]]))
# 8 points
    
"""
PART 2: logistic regression 
You are a recruiter and your goal is to predict whether an applicant is likely to get hired or rejected. 
You have gathered data over the years that you intend to use as a training set. 
Your task is to use logistic regression to build a model that predicts whether an applicant is likely to
be hired or not, based on the results of a first round of interview (which consisted of two technical questions).
The training instances consist of the two exam scores of each applicant, as well as the hiring decision.
"""

# Open the csv file in Excel, notepad++ or any other applications to have a rough overview of the data at hand. 

# Load the data from the file 'LogisticRegressionData.csv' in a pandas dataframe. Make sure all the instances 
# are imported properly. Name the first feature 'Score1', the second feature 'Score2', and the class 'y'
data = pandas.read_csv(r"C:\Users\sande\Desktop\final semester\machine learning\assignments\A2\LogisticRegressionData.csv", header = None, names=['Score1', 'Score2', 'y']) # 2 points

# Seperate the data features (score1 and Score2) from the class attribute 
X = data[["Score1", "Score2"]]# 2 points
y = data["y"]# 2 points

# Plot the data using a scatter plot to visualize the data. 
# Represent the instances with different markers of different colors based on the class labels.
m = ['o', 'x']
c = ['hotpink', '#88c999']
fig = plt.figure()
for i in range(len(data)):
    plt.scatter(X['Score1'][i], X['Score2'][i], marker=m[data['y'][i]], color = c[data['y'][i]]) # 2 points
fig.canvas.draw()

# Train a logistic regression classifier to predict the class labels y using the features X
regS = linear_model.LogisticRegression() # 2 points
regS.fit(X, y) # 2 points

# Now, we would like to visualize how well does the trained classifier perform on the training data
# Use the trained classifier on the training data to predict the class labels
y_pred = regS.predict(X) # 2 points
print(y_pred)
print(y)
# To visualize the classification error on the training instances, we will plot again the data. However, this time,
# the markers and colors selected will be determined using the predicted class labels
m = ['o', 'x']
c = ['red', 'blue'] #this time in red and blue
fig = plt.figure()
for i in range(len(data)):
    plt.scatter(X['Score1'][i], X['Score2'][i], marker=m[y_pred[i]], color= c[y_pred[i]]) # 2 points
fig.canvas.draw()
# Notice that some of the training instances are not correctly classified. These are the training errors.

"""
PART 3: Multi-class classification using logistic regression 
Not all classification algorithms can support multi-class classification (classification tasks with more than two classes).
Logistic Regression was designed for binary classification.
One approach to alleviate this shortcoming, is to split the dataset into multiple binary classification datasets 
and fit a binary classification model on each. 
Two different examples of this approach are the One-vs-Rest and One-vs-One strategies.
"""

#  One-vs-Rest method (a.k.a. One-vs-All)

# Explain below how the One-vs-Rest method works for multi-class classification # 12 points
"""
Machine learning models such as logistic regression are able to handle binary classification datasets but using some heuristic methods
we are able to handle multiclass classification problems as well. One such heuristic method used for solving multi-class classification problem is the
One-vs-Rest method and it works by determining the number of classifiers required to solve the problem. Usually for binary classification, the number of 
classifiers is equal to 1. However, if we use the One vs Rest method, we require N number of classifiers to solve a particular problem. This N number of 
classifiers is determined by the number of target variables. For each of these N classifiers, we convert the problem to binary classification [1]. 
These N number of classifiers is equivalent to the number of target values. For example, the dataset used for question number 4 has 3 class 
labels : galaxies, quasars, and stars. It is a multiclass classification problem and if the One-vs-Rest method is used to solve this problem then we 
will require 3 classifiers. For the first classifier we will use the target value galaxies and label it as class 1 or positive class and the other two class values 
will be labelled as 0 or the negative class. Similarly, for the second classifier the quasars target class will be used as positive class and the other two will 
belong to the negative class label. The third classifier will use the last class label which is the star class as positive class and the rest as negative class. 
As the name suggests One-vs-Rest method will use one class label versus the other class labels to build a classifier that separates it from the other classes and these will be 
different binary classifiers trained to classify one class versus the other classes. In the example discussed, there will be a total of 3 binary classifiers. 

"""

# Explain below how the One-Vs-One method works for multi-class classification # 11 points
"""
The One-vs-One method is another heuristic method that can be used to allow classical binary classification algorithms to perform multi-class classification. 
It works by generating several binary classifiers using the following formula (N*(N-1)/2), where N is the number of target values [1]. Using the same example discussed
above, we will end up having 3 classifiers which is the same number of classifiers required if we use the One-vs-Rest method. The only difference is the way it separates 
the target values in order to build the classifiers. In this case we use pair of classes to build binary classifiers to solve the multi-class classification problem. 
For example, to classify stars, quasars, and galaxies, there will be 3 binary classifiers. The first classifier will classify quasars from stars. 
The second classifier will identify stars from galaxies and the last classifier will learn to separate quasars from galaxies. 
These methods are really simple to understand in theory but in practice if the training data is complex it might not be conventional to use these methods as they might require 
more compute power. The application of these methods is really simple and can be done in a few lines of code by using the libraries and packages provided by sklearn as 
shown in question #4. 

References:
[1] A. Band, “Multi-class classification — one-vs-all &amp; one-vs-one,” Multi-class Classification — One-vs-All &amp; One-vs-One, 09-May-2020. Available: https://towardsdatascience.com/multi-class-classification-one-vs-all-one-vs-one-94daed32a87b. 
"""


############## FOR GRADUATE STUDENTS ONLY (the students enrolled in CPS 8318) ##############
""" 
PART 4 FOR GRADUATE STUDENTS ONLY: Multi-class classification using logistic regression project.
Please note that the grade for parts 1, 2, and 3 counts for 70% of your total grade. The following
work requires you to work on a project of your own and will account for the remaining 30% of your grade.

Choose a multi-Class Classification problem with a dataset (with a reasonable size) 
from one of the following sources (other sources are also possible, e.g., Kaggle):

•	UCI Machine Learning Repository, https://archive.ics.uci.edu/ml/datasets.php. 

•	KDD Cup challenges, http://www.kdd.org/kdd-cup.


Download the data, read the description, and use a logistic regression approach to solve a 
classification problem as best as you can. 
Investigate how the One-vs-Rest and One-vs-One methods can help with solving your problem.
Write up a report of approximately 2 pages, double spaced, in which you briefly describe 
the dataset (e.g., the size – number of instances and number of attributes, 
what type of data, source), the problem, the approaches that you tried and the results. 
You can use any appropriate libraries. 


Marking: Part 4 accounts for 30% of your final grade. In the write-up, cite the sources of 
your data and ideas, and use your own words to express your thoughts. 
If you have to use someone else's words or close to them, use quotes and a citation.  
The citation is a number in brackets (like [1]) that refers to a similar number in the references section 
at the end of your paper or in a footnote, where the source is given as an author, title, URL or 
journal/conference/book reference. Grammar is important. 

Submit the python script (.py file(s)) with your redacted document (PDF file) on the D2L site. 
If the dataset is not in the public domain, you also need to submit the data file. 
Name your documents appropriately:
report_Firstname_LastName.pdf
script_ Firstname_LastName.py
"""


#import all libraries

import numpy as np
import pandas
import pandas as pd
from sklearn import datasets 
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import seaborn as sns

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


#dataset load
df=pandas.read_csv(r"C:\Users\sande\Desktop\final semester\machine learning\assignments\A2\archive (25)\star_classification.csv")
df.head()

df.info()

df.isnull().sum() #there are no null attributes

df.shape

df["class"].value_counts()


from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
df["class"]=label_encoder.fit_transform(df["class"])
df.head()

df.dtypes
df["class"].unique()


plt.figure(figsize=(20,20))
df.hist('class')
plt.show('class')


sns.displot(df, x="field_ID")

#undersampling to handle imbalanced data
#divide by class
df_class_0=df[df['class']==0]
df_class_1=df[df['class']==1]
df_class_2=df[df['class']==2]


count_class_0, count_class_2, count_class_1= df["class"].value_counts()
df_class_0_under=df_class_0.sample(count_class_1)
df_class_2_under=df_class_2.sample(count_class_1)

newdf=pd.concat([df_class_0_under,df_class_2_under,df_class_1], axis=0)
#oversampling to handle imbalanced data
newdf["class"].value_counts()



#scale data and feature reduction
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
#"obj_ID", "run_ID", "rerun_ID", "field_ID", "fiber_ID",
X=newdf.drop(["obj_ID", "run_ID", "rerun_ID", "field_ID", "fiber_ID", "spec_obj_ID","class"],  axis=1)
y=newdf['class']


X[["alpha", "delta", "u", "g", "r", "i", "z", "cam_col", "redshift", "plate", "MJD"]]=scaler.fit_transform(X[["alpha", "delta", "u", "g", "r", "i", "z", "cam_col", "redshift", "plate", "MJD"]])
X


#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=25, shuffle=True)
 
 
 
#build and train model one vs rest
model=LogisticRegression()
ovr=OneVsRestClassifier(model)
clf=ovr.fit(X_train,y_train)

y_pred= clf.predict(X_test)
print("The accuracy of the classifier is", accuracy_score(y_test, y_pred))


print("The accuracy of the classifier is", accuracy_score(y_test, y_pred))


print("The recall of the classifier is", recall_score(y_test, y_pred, average= 'weighted'))

print("The precision of the classifier is", precision_score(y_test, y_pred, average= 'weighted'))

print("The f1 score of the classifier is", f1_score(y_test, y_pred, average= 'weighted'))

cm= confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)


#build and train model one vs. one
model=LogisticRegression()
ovo=OneVsOneClassifier(model)
clf=ovo.fit(X_train,y_train)

y_pred= clf.predict(X_test)
print("The accuracy of the classifier is", accuracy_score(y_test, y_pred))


print("The recall of the classifier is", recall_score(y_test, y_pred, average= 'weighted'))

print("The precision of the classifier is", precision_score(y_test, y_pred, average= 'weighted'))

print("The f1 score of the classifier is", f1_score(y_test, y_pred, average= 'weighted'))

cm= confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True)
 


