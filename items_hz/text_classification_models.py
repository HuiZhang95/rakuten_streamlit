import streamlit as st
from PIL import Image
import pickle
import numpy as np

def text_classification_models():
    
    st.markdown("<h3>Customized recurrent neural network (RNN)</h3>", unsafe_allow_html = True)
    with st.expander("click here for details"):
        st.markdown("<h3>Background + procedure</h3>", unsafe_allow_html = True)

        st.write("Recurrent layers are designed to capture sequential relationships within the data."
                "This is especially important for the text information in the present project. "
                " \n\n"
                "Procedure: \n "
                "Train-test split --> tokenize sequences --> train model"
                "Here is the achitechture of the model:")
        
        img = Image.open("images_hz/RNN achitechture.jpg")
        st.image(img, use_container_width = True)

        st.markdown("<h3>Training process</h3>", unsafe_allow_html = True)
        img = Image.open("images_hz/RNN training.jpg")
        st.image(img, use_container_width = True)

        st.markdown("<h3>Results</h3>", unsafe_allow_html = True)
        st.write("Weighted F1 score is 0.75")
        img = Image.open("images_hz/RNN result.jpg")
        st.image(img, use_container_width = True)

        img = Image.open("images_hz/RNN result 2.jpg")
        st.image(img, use_container_width = True)

    st.markdown("<h3>SVM with gridsearch</h3>", unsafe_allow_html = True)
    with st.expander("click here for details"):
        st.markdown("<h3>Background + procedure (RNN)</h3>", unsafe_allow_html = True)

        st.write("SVM aims to identify decision boundaries that maximize the margin between categories,"
                 " which helps generalization on unseen data. "
                 "Additionally, SVM uses a subset of training points to define the hyperplane, "
                 "making it memory efficient. This is particularly beneficial for systems with limited resources.. "
                " \n\n"
                "Procedure: \n "
                "TF-IDF transformation --> search for optimal hyperparameters"
                "Here is the gridsearch result:")
        
        img = Image.open("images_hz/SVM model.jpg")
        st.image(img, use_container_width = True)

        st.markdown("<h3>Results</h3>", unsafe_allow_html = True)
        st.write("Weighted F1 score is 0.81")
        img = Image.open("images_hz/SVM result.jpg")
        st.image(img, use_container_width = True)

    # use the st.selectbox() method to choose between the RandomForest classifier, the SVM classifier and the LogisticRegression classifier. Then return to the Streamlit web application to view the select box.
    choice = ['RNN', 'SVC']
    option = st.selectbox('Choice of the model', choice)
    st.write('The chosen model is :', option)


    
        

