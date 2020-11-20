import streamlit as st
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os

def file_selector(folder_path='./dataset'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox("Select A file",filenames)
    return os.path.join(folder_path,selected_filename)

filename = file_selector()
st.info("You Selected {}".format(filename))

# Read Data
df = pd.read_csv(filename)
# Show Dataset


#cleaning the data

df.dropna(axis=1, how='all')
  
if st.checkbox("Show Dataset"):
    st.dataframe(df)
   
    
st.subheader("Data Visualization")
  # Correlation
  # Seaborn Plot
st.set_option('deprecation.showPyplotGlobalUse', False)
if st.checkbox("Correlation Plot[Seaborn]"):
    st.write(sns.heatmap(df.corr(),annot=True))
    st.pyplot()

#encoding the things // data transformation
label= LabelEncoder()
for col in df.columns:
    df[col]=label.fit_transform(df[col])
Y = df.target
X = df.drop(columns=['target'])
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.33, random_state=8)

#models
classifier_name = st.selectbox(
      'Machine Learning Algorithm',
      ('Linear Regression', 'SVR','Lasso Regression','Decision Tree')
  )

#if model is Linear Regression
if classifier_name == 'Linear Regression':
    if st.button("classify",key='classify'):
        st.subheader("Linear Regression result")
        linear= LinearRegression()
        linear.fit(X_train,y_train)
        preds= linear.predict(X_test)
        st.write("R2 score : %.2f" % r2_score(y_test,preds))
        st.write("Mean squared error: %.2f" % mean_squared_error(y_test,preds))
#if model is svr
if classifier_name == 'SVR':
    if st.button("classify",key='classify'):
        st.subheader("SVR result")
        svrmodel= SVR()
        svrmodel.fit(X_train,y_train)
        preds= svrmodel.predict(X_test)
        st.write("R2 score : %.2f" % r2_score(y_test,preds))
        st.write("Mean squared error: %.2f" % mean_squared_error(y_test,preds))
              

#if model is Lasso
if classifier_name == 'Lasso Regression':
    if st.button("classify",key='classify'):
        st.subheader("Lasso Regression result")
        lassomodel= Lasso()
        lassomodel.fit(X_train,y_train)
        preds= lassomodel.predict(X_test)
        st.write("R2 score : %.2f" % r2_score(y_test,preds))
        st.write("Mean squared error: %.2f" % mean_squared_error(y_test,preds))

#if model is decision
if classifier_name == 'Decision Tree':
    if st.button("classify",key='classify'):
        st.subheader("Decision  result")
        decisionModel= DecisionTreeRegressor()
        decisionModel.fit(X_train,y_train)
        preds= decisionModel.predict(X_test)
        st.write("R2 score : %.2f" % r2_score(y_test,preds))
        st.write("Mean squared error: %.2f" % mean_squared_error(y_test,preds))
              
                        
              