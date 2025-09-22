# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary libraries like 'pandas' for data manipulation and 'numpy' for numerical operations.
2. Load the CSV file ('Placement_Data.csv') into a pandas DataFrame and preview the first 5 rows using 'df.head()'.
3. Make a copy of the original DataFrame ('df') to preserve the original data.
4. Drop the columns 'sl_no' (serial number) and 'salary' because they are not required for modeling.
5. '.isnull().sum()' returns the count of missing values for each column.
6. '.duplicated().sum()' returns the count of duplicate rows in the dataset.
7. The 'LabelEncoder' from 'sklearn' is used to transform string labels into numeric labels.
8. This is done for columns 'gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation', and 'status' where each unique category is assigned a numeric value.
9. 'x' is created by selecting all columns except for the last column ('status'), which is the target variable.
10. 'y' is the 'status' column, which represents whether the student got placed or not (binary classification).
11. The data is split using 'train_test_split' from 'sklearn'.
12. 'test_size=0.2' means that 20% of the data will be used for testing, and 80% will be used for training.
13. 'random_state=0' ensures that the split is reproducible.
14. 'LogisticRegression(solver="liblinear")' creates a logistic regression classifier using the 'liblinear' solver (good for smaller datasets).
15. 'lr.fit(x_train, y_train') trains the model on the training data.
16. Use the trained logistic regression model ('lr') to predict the target
17. values ('status') for the test set ('x_test').
18. 'accuracy_score' compares the predicted values ('y_pred') with the true values ('y_test') and calculates the accuracy of the model.
19. 'confusion_matrix(y_test, y_pred)' outputs the confusion matrix based on the true values and predicted values.
20. 'classification_report(y_test, y_pred)' provides metrics such as precision, recall, F1-score, and support for each class (in this case, whether a student got placed or not).
21. The input should match the features used in the model (numeric values representing different attributes of the student).

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:SWETHA S
RegisterNumber:212224040344

import pandas as pd
import numpy as np
df=pd.read_csv('Placement_Data.csv')
print("Name: SWETHA S\nReg.no: 212224040344)
df.head()

df1=df.copy()
df1=df1.drop(["sl_no","salary"],axis = 1)
df1.head()

df1.isnull().sum()

df1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df1['gender']=le.fit_transform(df1['gender'])
df1['ssc_b']=le.fit_transform(df1['ssc_b'])
df1['hsc_b']=le.fit_transform(df1['hsc_b'])
df1['hsc_s']=le.fit_transform(df1['hsc_s'])
df1['degree_t']=le.fit_transform(df1['degree_t'])
df1['workex']=le.fit_transform(df1['workex'])
df1['specialisation']=le.fit_transform(df1['specialisation'])
df1['status']=le.fit_transform(df1['status'])
df1

x=df1.iloc[:,:-1]
x

y=df1['status']
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report=classification_report(y_test,y_pred)
print(classification_report)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

*/  
```

## Output:
<img width="1157" height="272" alt="image" src="https://github.com/user-attachments/assets/5b043d5c-ea27-4b85-8557-ea0825e10dc2" />



<img width="1025" height="202" alt="image" src="https://github.com/user-attachments/assets/b725e755-548b-45cf-8c11-954c231df69b" />



<img width="213" height="316" alt="image" src="https://github.com/user-attachments/assets/eb9a5277-58a3-4f0d-ba7a-4bc56cfb88a1" />



<img width="34" height="30" alt="image" src="https://github.com/user-attachments/assets/7c16220c-8d1b-46cf-887e-353e7cfca476" />



<img width="935" height="427" alt="image" src="https://github.com/user-attachments/assets/aa50bdf2-e870-48cb-be1c-2e924f722e31" />



<img width="880" height="435" alt="image" src="https://github.com/user-attachments/assets/16bb7f95-d4bb-4c59-b4a7-0bfc7cbca2c6" />



<img width="388" height="263" alt="image" src="https://github.com/user-attachments/assets/d8a2a6c2-6256-406c-8a0e-99014e37c606" />



<img width="702" height="61" alt="image" src="https://github.com/user-attachments/assets/30914915-d1e1-4f0b-86aa-18f93fc8c91c" />



<img width="184" height="34" alt="image" src="https://github.com/user-attachments/assets/fc4a4fa8-f5a9-4539-9d3f-21e16d5e6a02" />



<img width="298" height="67" alt="image" src="https://github.com/user-attachments/assets/e048f67c-22fa-4c0f-88bf-0ca161b69594" />



<img width="529" height="186" alt="image" src="https://github.com/user-attachments/assets/7e36cbaa-2f18-4c8f-89fb-44e66acc6dc0" />



<img width="106" height="31" alt="image" src="https://github.com/user-attachments/assets/3fba8998-f9da-4d55-ac47-ca6e1c572d8c" />










## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
