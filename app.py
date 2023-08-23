import numpy as np
import pickle
import pandas as pd
import streamlit as st 
from PIL import Image

pickle_in = open("RF copy.pkl","rb")
classifier=pickle.load(pickle_in)



def welcome():
    return "Welcome All"


def predict_note_authentication(Nitrogen,Phosphorus,Pottassium,Temperature,Humidity,PH_Value,Rainfall):
    
   
    prediction=classifier.predict([[Nitrogen,Phosphorus,Pottassium,Temperature,Humidity,PH_Value,Rainfall]])
    print(prediction)
    return prediction


def main():

    st.set_page_config(
        page_title="Crop Recommendation System",
        page_icon="Tutorials.png",
        layout="wide",
    )


    image = Image.open('crop rec image.png')
    st.image(image)

    st.title("Crop Recommendation System using Machine Learning ")
    html_temp = """
    <div style="background-color:tomato;padding:0px">
    <h3 style="color:white;text-align:center;"> Enter the Values to get the predictions    </h3>
    </div>
    <br>
    """

    st.markdown('Our app predicts the best crop that fits to the given **FARM LAND**.')
    st.markdown('Among all the 7 various Algorithms **Random Forest** has the best accuracy.')

    st.markdown(html_temp,unsafe_allow_html=True)
    Nitrogen = st.text_input("Nitrogen",placeholder="Enter the value of Nitrogen")
    Phosphorus = st.text_input("Phosphorus",placeholder="Enter the value of Phosphorus")
    Pottassium = st.text_input("Pottassium",placeholder="Enter the value of Pottassium")
    Temperature = st.text_input("Temperature",placeholder="Enter the value of Temperature")
    Humidity = st.text_input("Humidity",placeholder="Enter the value of Humidity")
    PH_Value = st.text_input("PH_Value",placeholder="Enter the value of PH")
    Rainfall = st.text_input("Rainfall",placeholder="Enter the value of Rain Fall")

    result=""
    if st.button("Predict the Crop"):
        result=predict_note_authentication(Nitrogen,Phosphorus,Pottassium,Temperature,Humidity,PH_Value,Rainfall)
    st.success('The crop that fits to the farm land is :   {}'.format(result))
    if st.button("Click here to get Dataset"):
        with open('Crop_Recommendation_System_Data.csv') as f:
            st.download_button('Download CSV', f,"Crop Recommend_Dataset/csv") 
            
             
        
    

    

if __name__=='__main__':
    main()
    
    
    
