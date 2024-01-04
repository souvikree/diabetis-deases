
import pandas as pd;
import numpy as np;
df=pd.read_csv("diabetes.csv")
x=df.iloc[:,0:8].values
y=df.iloc[:,8].values

#Splitting the data set to form training and testing data.
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#Applying K-NN machine algorithm to train the model
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(weights='distance')
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)

#Getting the confusion matrix ready.
from sklearn.metrics import confusion_matrix
print("This is my confusion matrix :"
      ,confusion_matrix(y_test, y_pred))

#Getting the accuracy score ready.
from sklearn.metrics import accuracy_score
print("This is the my model using KNN:")
print(accuracy_score(y_test, y_pred))

#Applying the Naive Bayes method to train my model
from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(x_train,y_train)
y_pred2=gnb.predict(x_test)

#Getting the confusion matrix ready.
from sklearn.metrics import confusion_matrix
print("This is my confusion matrix :"
      ,confusion_matrix(y_test, y_pred2))

#Getting the accuracy score ready.
from sklearn.metrics import accuracy_score
print("This is the my model using naive-Bayes:")
print(accuracy_score(y_test, y_pred2))

#Applying the SVM Classifier
from sklearn.svm import SVC
sv=SVC(kernel='linear',random_state=0)
sv.fit(x_train,y_train)
y_pred3=sv.predict(x_test)

#Getting the confusion matrix ready.
from sklearn.metrics import confusion_matrix
print("This is my confusion matrix :"
      ,confusion_matrix(y_test, y_pred3))

#Getting the accuracy score ready.
from sklearn.metrics import accuracy_score
print("SVM Accuracy Score:")
print(accuracy_score(y_test, y_pred3))

#Building a predictive system for diabetes prediction.
input_data=(2,197,70,45,543,30.5,0.158,53)
#changing the input data to a numpy array.
input_data_array=np.asarray(input_data);
id_reshaped=input_data_array.reshape(1,-1);
std_data=sc.transform(id_reshaped);
print(std_data)
#Getting the predictor ready.
prediction=sv.predict(std_data)
if(prediction[0]==1):
    print("This person is diabetic!!!");
else:
    print("This person is non diabetic!!!");