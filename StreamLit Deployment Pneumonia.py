#!/usr/bin/env python
# coding: utf-8

# In[18]:


#pip install streamlit


# In[11]:


import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image


# In[16]:


model = tf.keras.models.load_model("Pneumonia xray.keras")


# In[17]:


st.title("Pneumonia Detection from Chest X-Ray")
st.write("Upload a chest X-ray image to predict whether it shows sign of pneumonia.")

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width= True)

    #preprocessing
    img = img.resize((32,32))
    img_array = image.img_to_array(img) /255.0
    img_array = np.expand_dims(img_array,axis=0)

    #Prediction
    prediction = model.predict(img_array)
    confidence = float(prediction[0][0])

    #output
    if confidence > 0.5:
        st.error("Prediction: Pneumonia")
    else:
        st.success("Prediction: Normal")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




