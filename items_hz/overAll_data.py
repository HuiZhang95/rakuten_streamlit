import streamlit as st
from PIL import Image


def overAll_data():
    
    st.markdown("<h2>Data description</h2>", unsafe_allow_html = True)

    img = Image.open("images/overall_data.png")
    st.image(img, use_container_width = True)

    st.write("""The text consists of 84,916 product codes ('prdtypecode') which are 
             associated with two columns of descriptive text about the products, 
             'designation’ and 'description'. The images consist of 84,914 single images 
             in .jpg format as a visual representation of the product.
             """)

    st.write("""There are 27 different product categories.
             The distribution of the product categories shows that the dataset has an unbalanced representation of the individual categories.
             Due to the nature of our dataset, we will use several methods to eliminate bias caused by unbalanced distribution across categories, including the weighted F1-score as a metric for the performance of our classification models, as well as weighted classes.
             """)
