import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

def image_description():

    st.markdown("<h2>Data Description</h2>", unsafe_allow_html = True)

    st.write("The Rakuten Project includes over 80,000 images of different products "
             "across 27 different categories. In this section, we present an overview "
             "of the products by category, a description of their distribution, and "
             "the data processing we implemented. Below is a random sample of images "
             "representing each category.")

    with st.expander("Click to view randomly sampled images for each category"):
        img = Image.open("images/random_sample.png")
        st.image(img, use_container_width = True)

    st.write("We found that approximately 6.7% of the data - or 5,692 images - contained duplicates. "
             "Of those, there were 3,264 images that had at least one duplicate. Duplicate images "
             "typically belonged to the same category, but we found that 73 images belonged to multiple "
             "categories. Here, we visualize the distribution of the duplicates with mutliple "
             "categories and the distribution of duplicates across categories.")

    with st.expander("Click to view random sample of duplicates and their categories"):
        img = Image.open("images/sample_duplicates.png")
        st.image(img, use_container_width = False)

    with st.expander("Click to view the distribution of duplicates across categories"):
        img = Image.open("images/duplicates_categories.png")
        st.image(img, use_container_width = False)

    st.write("We also found significant distribution in the size of the images. That is to say that "
             "while images were all 500 x 500, many of the images had a significant amount of white "
             "padding around the images. In the plot below, the the ratio of the image is referenced "
             "as image_size/500 x 500. Therefore, images with a ratio of 1 are 500 x 500 (i.e., no "
             "white padding), while images smaller than that contain white padding.")

    with st.expander("Click to view the distribution of image sizes"):
        img = Image.open("images/image_sizes.png")
        st.image(img, use_container_width = False)

    st.write("In order to maximize the image ratios, we created a custom resizing function that "
             "trims the white space, and resizes the largest size of the image to 400 pixels, while "
             "maintaining the correct aspect ratio. We then added padding to the sides or top and bottom "
             "to ensure that each image was 400 x 400 pixels after resizing. The below plot demonstrates "
             "what area of the plot was registered as 'white space' in order to calculate the bbox "
             "and then what the image looks like after resizing.")

    with st.expander("Click to view the distribution of image sizes"):
        img = Image.open("images/resized_example.png")
        st.image(img, use_container_width = False)

    st.markdown("<h2>Data Visualization</h2>", unsafe_allow_html = True)    

    st.write("We visualized the data to better understand distributions using a "
             "variety of methods, including LLE, UMAP Project, PCA, and TSNE.")

    st.markdown("<h3>Locally Linear Embedding Plot</h3>", unsafe_allow_html = True)

    with st.expander("Click to view Locally Linear Embedding plot"):
        img = Image.open("images/LLE.png")
        st.image(img, use_container_width = True)

    st.markdown("<h3>UMAP Projection Plot</h3>", unsafe_allow_html = True)

    with st.expander("Click to view UMAP Projection plot"):
        img = Image.open("images/UMAP.png")
        st.image(img, use_container_width = False)

    st.markdown("<h3>PCA Plot</h3>", unsafe_allow_html = True)

    with st.expander("Click to view PCA plot"):
        img = Image.open("images/PCA.png")
        st.image(img, use_container_width = False)

    st.markdown("<h3>TSNE Plot</h3>", unsafe_allow_html = True)

    with st.expander("Click to view TSNE plot"):
        img = Image.open("images/TSNE.png")
        st.image(img, use_container_width = False)

    st.markdown("The plots of the reduced feature space using various methods did not reveal "
                "well defined clusters in the dataset. This suggests the differences between "
                "the images across categories are complex in nature and will require a "
                "complex classifying solution.")
    
                     

    

  
