import streamlit as st
from PIL import Image


def ensemble_model():
    
    st.markdown("<h3>Feature fusion approach</h3>", unsafe_allow_html = True)
    with st.expander("click here for details"):
        st.markdown("<h3>Background + procedure</h3>", unsafe_allow_html = True)

        st.write("Data from various modalities were processed using different models. "
                 "In this study, text data were trained with a pre-trained RoBERTa model, "
                 "while image data were trained using a custom ResNet50 model, EfficientNetV2 model, "
                 "and a ViT model. "
                 ""
                 "Procedure:"  
                 "Step 1: extract intermediate features from the deepest layer of the trained models "
                 "      prior to the classifier head. "
                 "      2815 features for images and 2407 features for text "
                 "Step 2: scale text and image features separately to a range from -1 to 1. "
                 "Step 3: concatenated text and image features. "
                 "Step 4: train a SVM with gridsearch")
        
        img = Image.open("images_hz/feature_fusion procedure.jpg")
        st.image(img, use_container_width = True)

        st.markdown("<h3>Results</h3>", unsafe_allow_html = True)
        st.write("Weighted F1 score is 0.75")
        img = Image.open("images_hz/feature_fusion result.jpg")
        st.image(img, use_container_width = True)

        img = Image.open("images_hz/feature_fusion comparing models.jpg")
        st.image(img, use_container_width = True)
