import streamlit as st

def text_classification_models():
    
    with st.expander("<h2>Customized recurrent neural network (RNN)</h2>", unsafe_allow_html = True):
        st.markdown("<h3>Background + procedure (RNN)</h3>", unsafe_allow_html = True)

        st.write("Recurrent layers are designed to capture sequential relationships within the data."
                "This is especially important for the text information in the present project. "
                " "
                "Procedure: Train-test split --> tokenize sequences --> train model"
                "Here is the achitechture of the model:")
        
        with st.expander("RNN achitechture"):
            img = Image.open("images/RNN achitechture.jpg")
            st.image(img, use_container_width = True)

        st.write("For this project, we have use the three pre-trained models as the backbone "
                "for our model architecture, ResNet50 (He et al., 2016), EfficientNetV2 "
                "(Tan & Le, 2021), and ViT (Dosovitskiy et al., 2021).")
