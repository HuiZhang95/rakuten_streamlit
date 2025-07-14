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


    st.markdown("<h3>roBERTa-base</h3>", unsafe_allow_html = True)
    with st.expander("click here for details"):
        st.markdown("<h3>Background + procedure (roBERTa-base)</h3>", unsafe_allow_html = True)

        st.write("""Our first model was roBERTa-base, trained only on English
             data. We used the translated text and implemented a finally
             formatting of the data using spacy. We tokenized, lemmatized,
             removed stopwords and punctuation, and any unnecessary
             white space, and finally converted words to lower case. We
             then began model training. The tokenizer had a length of
             128. We used a batch size of 64 due to hardware limitations.
             We froze all the layers of the roBERTa model, except the final
             5 layers. We used an Adam optimizer and we set the starting
             learning rate at 1e-4, due to benefits of a slower rate when
             training deeper layers. However, we did also experiment with
             1e-3 and 1e-5 starting rates. We used a learning rate scheduler
             with a patience of 3 epochs and a reduction factor of 0.1. We
             also computed class weights using scikit-learn's
             compute_class_weights function, with the 'balanced' class_weight
             setting. We monitored training and validation F1-Score and
             Loss and the training gradients. We utilized an early stopping
             function that would restore the model with the best F1-Score if
             no reduction in loss was seen after 6 epochs or if no reduction
             in F1-Score was seen in 10 epochs. Early stopping triggered at
             9 epoch due to a gradually increasing loss. Even though there
             was a slight increase in F1-Score over time, this stop was
             necessary to prevent severe overfitting. The model found
             meaningful connections between target words. For example,
             for Class 60, the model found a meaningful connection with
             words such as 'Playstation' and 'Sony' and class membership.
             Below we present an image of the training history and an
             example of feature imporance extraction for a single class.""")
        
        img = Image.open("images/roBERTa_training_history.png")
        st.image(img, use_container_width = True)
        img = Image.open("images/mean_attention_map_roBERTa_class_60.png")
        st.image(img, use_container_width = True)


    

    # use the st.selectbox() method to choose between the RandomForest classifier, the SVM classifier and the LogisticRegression classifier.
    #Then return to the Streamlit web application to view the select box.
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


    
        

