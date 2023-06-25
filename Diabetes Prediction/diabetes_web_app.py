import numpy as np
import streamlit as st
import pickle
loaded_model = pickle.load(open("E:/ML Ops/Project Folder/Diabetes Prediction/trained_model.sav",'rb'))

# create function for prediction
def diabetes_pred(input_data):
    #changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array to one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


    #do prediction
    prediction = loaded_model.predict(input_data_reshaped )
    res = ''
    if(prediction[0] == 0):
       res = "The person does not have diabetes"
    else:
       res = "The person has diabetes"
    return res


def main():
   
   # Give title 
   st.title("Diabetes Prediction Web App")

   # Getting the input data-set
   # Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
   Pregnancies = st.text_input('Number of Pregnancies')
   Glucose = st.text_input('Glucose Level')
   BloodPressure = st.text_input('BloodPressure Value')
   SkinThickness = st.text_input('SkinThickness Value')
   Insulin = st.text_input('Insulin Level')
   BMI = st.text_input('BMI')
   DiabetesPedigreeFunction= st.text_input('NDiabetesPedigreeFunction value')
   Age = st.text_input('Age')

   # code for prediction
   diagnosis = ''

   # create button for prediction
   if st.button('Diabetes Test Result'):

      diagnosis = diabetes_pred([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])


   st.success(diagnosis)

if __name__ == '__main__':
   main()