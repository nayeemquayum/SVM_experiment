import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
#load data in bunch format
iris=load_iris()
#check the features of sklearn data set
print("iris features:",iris.feature_names)
print("irsi groups:",iris.target_names)
#create dataframe from bunch
iris_df=pd.DataFrame(iris.data,columns=iris.feature_names)
iris_df['species'] = iris.target
print ("iris:",iris_df.info())
iris_df.rename(columns={'sepal length (cm)' : 'sepal_length_cm',
                     'sepal width (cm)' : 'sepal_width_cm',
                     'petal length (cm)': 'petal_length_cm',
                     'petal width (cm)' : 'petal_width_cm'}, inplace=True)
print ("iris:",iris_df.info())
# sns.relplot(data=iris_df, x='sepal_length_cm', y='sepal_width_cm', kind='scatter',hue='species')
# plt.show()
# sns.relplot(data=iris_df, x='petal_length_cm', y='petal_width_cm', kind='scatter',hue='species')
# plt.show()
#split data in train and test
X_train, X_test, y_train, y_test = train_test_split(iris_df.drop(columns=['species']),
                                                    iris_df["species"], test_size=0.2,random_state=20)
#create the model
svc_model = SVC()
#check how the model is performing on training data
svc_model.fit(X_train, y_train)
print("SVC score for the model:",svc_model.score(X_train, y_train))
#run cross validation
X = iris_df.iloc[:,0:4]
y = iris_df.iloc[:,4]
mean_corss_validation_score= cross_val_score(svc_model, X, y, cv=10,scoring='accuracy').mean()
print("Mean corss validation score for accuracy of SVC:",mean_corss_validation_score)
#now predict y
y_prediction= svc_model.predict(X_test)
#find the accuracy of the prediction
from sklearn.metrics import accuracy_score,confusion_matrix
print("Accuracy of SVM model:",accuracy_score(y_test, y_prediction))
#confusion_matrix
confusion_matrix=pd.DataFrame(confusion_matrix(y_test,y_prediction),columns=list(range(0,3)))
print("Confusion matrix",confusion_matrix.head())
#recall_score,precision_score,f1_score
from sklearn.metrics import recall_score,precision_score,f1_score
print("-"*25,"SVC Metrics","-"*25)
print("Precision:  ",precision_score(y_test,y_prediction,average='weighted'))
print("Recall: ",recall_score(y_test,y_prediction,average='weighted'))
print("F1 score: ",f1_score(y_test,y_prediction,average='weighted'))
print("-"*80)
#Create a 2D data for iris data set with "petal_length_cm" and "petal_weidth_cm"
iris_df.drop(['sepal_length_cm', 'sepal_width_cm'], axis='columns', inplace=True)
print("After dropping sepal length and width the dataframe:",iris_df.info())
from mlxtend.plotting import plot_decision_regions
#split data in training and test set
X_2D_train,X_2D_test,y_2D_train,y_2D_test = train_test_split(iris_df.drop(columns=['species']),
                                               iris_df['species'],test_size=0.2,random_state=20)
svc_model_2D = SVC()
svc_model_2D.fit(X_2D_train,y_2D_train)
plot_decision_regions(X_2D_train.values, y_2D_train.values, svc_model_2D, legend=2)
# Adding axes annotations
plt.xlabel('petal length(cm)')
plt.xlabel('petal width(cm)')
plt.title('SVC on Iris data')
plt.show()
