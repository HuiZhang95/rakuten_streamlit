import streamlit as st
from PIL import Image
import pickle
import numpy as np
import pandas as pd

def text_classification_models():
    
    st.markdown("<h3>Customized Recurrent Neural Network (RNN)</h3>", unsafe_allow_html = True)
    with st.expander("click here for details"):
        st.markdown("<h3>Background + procedure</h3>", unsafe_allow_html = True)

        st.write("Recurrent layers are designed to capture sequential relationships within the data."
                "This is especially important for the text information in the present project. "
                " \n\n"
                "Procedure: \n "
                "Train-test split --> tokenize sequences --> train model"
                "Here is the achitechture of the model:")
        
        img = Image.open("images/RNN achitechture.jpg")
        st.image(img, use_container_width = True)

        st.markdown("<h3>Training process</h3>", unsafe_allow_html = True)
        img = Image.open("images/RNN training.jpg")
        st.image(img, use_container_width = True)

        st.markdown("<h3>Results</h3>", unsafe_allow_html = True)
        st.write("Weighted F1 score is 0.75")
        img = Image.open("images/RNN result.jpg")
        st.image(img, use_container_width = True)

        img = Image.open("images/RNN result 2.jpg")
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
        
        img = Image.open("images/SVM model.jpg")
        st.image(img, use_container_width = True)

        st.markdown("<h3>Results</h3>", unsafe_allow_html = True)
        st.write("Weighted F1 score is 0.81")
        img = Image.open("images/SVM result.jpg")
        st.image(img, use_container_width = True)
        
    
    st.markdown("<h3>Pre-trained NLP Model: fastText</h3>", unsafe_allow_html = True)
    with st.expander("click here for details"):
        st.markdown("<h3>Background + procedure (fastText)</h3>", unsafe_allow_html = True)
        st.write("""fastText offers the possibility to represent words by vectors that can 
                 be used in modern machine learning models. The idea of these vectors is to 
                 capture hidden information about a language, like word analogies or semantics. 
                 Furthermore it offers the possibility to perform classification tasks in order 
                 to assign text to multiple classes. fastText is a quite efficient library and the 
                 adaptation to a new labeled dataset can be performed on a short time scale.
                 Bojanowski, 2017""")
        
        img = Image.open("images/image_01.png")
        st.image(img, use_container_width = True)

    st.markdown("<h3>Pre-trained NLP Model: roBERTa-large</h3>", unsafe_allow_html = True)
    with st.expander("click here for details"):
        st.markdown("<h3>Background + procedure (roBERTa-large)</h3>", unsafe_allow_html = True)
        st.write("""We have tried 'bert-base-uncased', 'roberta-large', and 'roberta-large'.
                 Within this project the model 'roberta-large' achieved the highest score among the tested models, therefore only the results from this model will be presented here.
                 max_length=64, padding=True, truncation=True. 
                 Liu, 2019
                 """)
        
        img = Image.open("images/image_04.png")
        st.image(img, use_container_width = True)
        img = Image.open("images/image_02.png")
        st.image(img, use_container_width = True)

    # use the st.selectbox() method to choose between the RandomForest classifier, the SVM classifier and the LogisticRegression classifier.
    #Then return to the Streamlit web application to view the select box.

    st.markdown("<h3>Time for game ;-)</h3>", unsafe_allow_html = True)
    with st.expander("click here for details"):

        df = pd.read_csv('{}rnn_svc_result.csv'.format('datasets/'), index_col = 0)
        len_df = len(df)
        index_df = df.index

        user_input_number = st.number_input("Input a number between 0 and {}: ".format(str(len_df)), 0)

        output_st = df.loc[index_df[user_input_number], 'designation']
        st.write('\nHere is the text information from designation column:\n', output_st)

        output_st = df.loc[index_df[user_input_number], 'text']
        st.write('\nHere is the text information from description column:\n', output_st)

        output_st = df.loc[index_df[user_input_number], 'text']
        st.write('\nHere is the text information after pre-processing:\n', output_st)
        
        choice = ['choose a model','RNN', 'SVC']
        option = st.selectbox('Choice of the model', choice)
        st.write('The chosen model is :', option)

        if option == 'drag and choose':
            st.write('Please choose a model.')

        if option == 'RNN':
            output_st = df.loc[index_df[user_input_number], 'y_RNN']
            st.write('The predicted category is: ', output_st)

            output_st = df.loc[index_df[user_input_number], 'y_true']
            st.write('The true category is: ', output_st)
        
        if option == 'SVC':
            output_st = df.loc[index_df[user_input_number], 'y_SVC']
            st.write('The predicted category is: ', output_st)

            output_st = df.loc[index_df[user_input_number], 'y_true']
            st.write('The true category is: ', output_st)


    
        

