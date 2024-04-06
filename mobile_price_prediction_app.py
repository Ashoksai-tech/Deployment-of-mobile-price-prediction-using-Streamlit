# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 19:14:08 2024

@author: aasho
"""

#load the saved model
import streamlit as st
import pickle
import numpy as np

loaded_model = pickle.load(open('C:/Users/aasho/OneDrive/Desktop/deploymodel/logreg.pkl','rb'))


def mobile_price_pred(input_data):

  input_data_as_array = np.asarray(input_data)

  input_data_reshaped = input_data_as_array.reshape(1, -1)

  prediction = loaded_model.predict(input_data_reshaped)
  print(prediction)

  if prediction[0] == 0:
      return 'lower end phone'

  elif prediction[0] == 1:
      return "little intermediate end phone"

  elif prediction[0] == 2:
      return "intermediate end phone"

  else:
      return "very lower end phone"



def main():


  st.title('Mobile Phone Price Predictor')

  # Input fields with more descriptive labels
  battery_power = st.text_input('Battery Power (mAh)')

  clock_speed = st.text_input('Clock Speed (GHz)')

  fc = st.text_input('Front Camera (MP)')

  int_memory = st.text_input('Internal Memory (GB)')

  m_dep = st.text_input('Mobile Depth (cm)')

  mobile_wt = st.text_input('Mobile Weight (g)')

  n_cores = st.text_input('Number of Cores')

  pc = st.text_input('Primary Camera (MP)')

  px_height = st.text_input('Pixel Height')

  px_width = st.text_input('Pixel Width')

  ram = st.text_input('RAM (GB)')

  sc_h = st.text_input('Screen Height (cm)')

  sc_w = st.text_input('Screen Width (cm)')

  talk_time = st.text_input('Talk Time (hours)')

  #code for prediction
  price_predictor = ''

  # Button to trigger prediction
  if st.button('Predict the price'):
    price_predictor = mobile_price_pred([battery_power,clock_speed,fc,int_memory,m_dep,
                                         mobile_wt,n_cores,pc,px_height,px_width,ram,sc_h,sc_w,talk_time])


  st.success(price_predictor)


if __name__=='__main__':
  main()
