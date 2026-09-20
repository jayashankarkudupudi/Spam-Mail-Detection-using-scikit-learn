import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.feature_extraction.text import CountVectorizer as cvt
from sklearn.naive_bayes import MultinomialNB as mnb
import streamlit as st

#reading the csv dataset as a pandas dataframe
data=pd.read_csv('spam.csv')

#Cleaning the dataset
data.drop_duplicates(inplace=True)
data['Category'] = data['Category'].replace(['ham','spam'],['Not Spam','Spam'])

#Categorising Splitting Training and Testing data
M = data['Message']
C = data['Category']

(M_train,M_test,C_train,C_test) = tts(M,C,test_size=0.2)

#Converting strings to numerical data using CountVectorizer
cv = cvt(stop_words='english')
features = cv.fit_transform(M_train)

#creating the model
model = mnb()
model.fit(features,C_train)

#testing the model
#features_test = cv.transform(M_test)
#print(model.score(features_test,C_test))
#Output:
#0.9806201550387597

#predicting the data
def pred(message):
    input_message = cv.transform([message]).toarray()
    result = model.predict(input_message)
    return result

#Creating a web UI 
st.header('Spam Detection')

user_message = st.text_input('Enter The Message Here')

if st.button('Confirm'):
    output = pred(user_message)
    st.markdown(output)
