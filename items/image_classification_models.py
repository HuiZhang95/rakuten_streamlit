import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

def image_classification_models():

    st.markdown("<h2>Convolutional Neural Networks</h2>", unsafe_allow_html = True)

    with st.expander("Click to see brief description of CNNs"):

        st.write("Convolutional neural networks (CNNs) are used extensively in image "
                 "recognition and machine vision models. These deep learning models "
                 "extract spatial heirarchies of features through backpropegation. "
                 "Models use several types of layers, including convolutional layers, "
                 "pooling layers, and fully connected layers in order to process images "
                 "for a variety of tasks, including classification (Yamashita et al., 2018). "
                 "Below is an example image of CNN architecture.")

        img = Image.open("images/cnn_example.jpg")
        st.image(img, use_container_width = True)

    st.write("For this project, we have use the three pre-trained models as the backbone "
             "for our model architecture, ResNet50 (He et al., 2016), EfficientNetV2 "
             "(Tan & Le, 2021), and ViT (Dosovitskiy et al., 2021).")

    st.markdown("<h2>ResNet</h2>", unsafe_allow_html = True)

    with st.expander("Click to see ResNet50 model description and results"):

        st.write("ResNet represented a breakthrough in CNNs. Prior to ResNet, a major issue "
                 "with 'very deep' architectures was that training and validation accuracy "
                 "was reduced. This has been described as the degredation problem. This issue "
                 "was not related to vanishing/exploding gradients as model architectures "
                 "incorporated normalizations which prevented that from occuring. He et al. (2016) "
                 "suggested that the degradation problem could be dealt with by incorporating "
                 "residual learning. Specifically, the authors introduced 'skip connections' "
                 "that explicitly implemented learning of residual features from the previous "
                 "convolutional layer.")

        img = Image.open("images/resnet50_skip_connections.png")
        st.image(img, use_container_width = True)

        st.write("In order to facilitate better classification of our dataset, we modified the "
                 "ResNet50 backbone slightly, by including additional convolutional layers, with "
                 "skip connections, and a more robust fully connected layer. Our goal was that "
                 "the additional convolutional layers would be trained specifically based on our "
                 "task, while the backbone that has been trained to extract important features "
                 "such as edges and shapes would remain mostly intact. To achive this, we "
                 "additionally froze all the ResNet50 layers aside from the last 3 bottlenecks. "
                 "The model included a total of 91,848,283 parameters, of which, 61,520,360 were "
                 "trainable.")

        st.markdown("<h3>Custom ResNet50 Results</h3>", unsafe_allow_html = True)

        st.write("The CustomResNet50 model used the base model as a backbone and features "
                 
                 "The CustomResNet50 new model weights were initialized using kaiming "
                 "initialization, with 'mode = fan_out'. We chose this method because "
                 "it is well suited for ReLU activation. We chose AdamW as the optimizer, "
                 "with betas = (0.9, 0.999), eps = 1e-8, weight_decay = 1e-4. The initial "
                 "learning rate = 1e-6. We used FocalLoss as the loss criterion, due to "
                 "the fact that it penalizes difficult to classify categories more than "
                 "easy to classify categories. This is especially valuable when you have "
                 "a dataset with relatively strong class imbalances, like we do. At Epoch "
                 "20, we increased the learning rate to 1e-3. At Epoch 30, we lowered the "
                 "unfrozen ResNet50 layers to 1e-5. At Epoch 38, we lowered the custom "
                 "layers learning rate to 1e-4. At Epoch 44 the early stoppage was triggered "
                 "and the model was reset to the best model based on F1-Score, which was "
                 "Epoch 43. Each of these events are marked on the plots below with dotted "
                 "vertical lines.")


        img = Image.open("images/custom_resnet50_results.png")
        st.image(img, use_container_width = False)

        st.write("During the training phase, we were able to achieve a weighted F1-Score "
                 "for the validation set of .60. We then evaluated the model on a final "
                 "test set and achieved a weighted F1-Score of .59. The full classification "
                 "report is below.")

        img = Image.open("images/classification_report_custom_resnet_50.png")
        st.image(img, use_container_width = False)

        st.write("We also extracted features for a subset of the images to better "
             "understand what the Custom ResNet50 model interpreted as important "
             "features.")

        img = Image.open("images/custom_gradcam.png")
        st.image(img, use_container_width = False)            


    st.markdown("<h2>EfficientNetV2</h2>", unsafe_allow_html = True)

    with st.expander("Click to see EfficientNetV2 model description and results"):

        st.write("EfficientNetV2 was designed to address the probelm that comes with training "
                 "increasingly large datasets. Namely, the amount of training required to "
                 "achieve a converged model systematically tends to increase as there are more "
                 "parameters and data to be trained on. To overcome these problems, EfficientNetV2 "
                 "uses training-aware neural architecture search and scaling, to jointly optimize "
                 "training speed and parameter efficiency. To optimize training, EfficientNetV2 "
                 "uses a combination of more commonly used depthwise convolutional 3 x 3 and "
                 "1 x 1 layers with fused convolutional layers (Tan & Le, 2021).")

        images = ['efficientnet_training_times', 'efficientnet_fused_layers']
        titles = ['Training Time Comparisons', 'Standard vs. Fused Layers']

        fig, ax = plt.subplots(1, 2)

        for index, axes in enumerate(ax):
            img = Image.open(f"images/{images[index]}.png")
            axes.imshow(img)
            axes.axis("off")
            axes.set_title(f"{titles[index]}")

        st.pyplot(fig, use_container_width = True)

        st.markdown("<h3>EfficientNetV2 Results</h3>", unsafe_allow_html = True)

        st.write("During the training process, we unfroze the lower layers (Block 6.0) of the model "
                 "in order to maintain the feature extraction trained on the initial layers and focus "
                 "computational power on adjusting the weights of the final layers to achieve "
                 "better task specific classification. The model included 10,737,731 parameters, of "
                 "which, 3,918,613 were trainable. We initiated training with an Adam optimizer "
                 "set with a learning rate of 1e-4 and weight decay of 1e-4. We used Focal Loss "
                 "as the criterion because of the inbalanced nature of the dataset. During training "
                 "we slowed the learning rate to 1e-5 after 10 epochs. The change in learning rate "
                 "is marked on the plot below with a dotted line.")

        img = Image.open("images/efficientnet_results.png")
        st.image(img, use_container_width = False)

        st.write("During the training phase, we were able to achieve a weighted F1-Score "
                 "for the validation set of .60. We then evaluated the model on a final "
                 "test set and achieved a weighted F1-Score of .60. The full classification "
                 "report is below.")

        img = Image.open("images/classification_report_efficientnet.png")
        st.image(img, use_container_width = False)

        st.write("We also extracted features for a subset of the images to better "
                 "understand what the Custom ResNet50 model interpreted as important "
                 "features.")

        img = Image.open("images/efficientnet_gradcam.png")
        st.image(img, use_container_width = False) 

    st.markdown("<h2>ViT</h2>", unsafe_allow_html = True)

    with st.expander("Click to see ViT model description and results"):

        st.write("We additionally trained a transformer, ViT (Dosovitskiy et al., 2021), as "
                     "part of our project development. Transformers are the standard for training "
                     "and developing large language models, but until recently have not been used "
                     "with machine vision. To achive this, preset 'patch sizes' of images are sent "
                     "through a transformer, where a learnable token 'classification' token is "
                     "added to each sequence.")

        img = Image.open("images/vit_architecture.png")
        st.image(img, use_container_width = True)
            
        st.write("As with the other models, we unfroze the lower layers, blocks 10 and 11, "
                     "to adjust the weights to task specific features, while freezing the "
                     "initial layers that are designed to capture broader, task invariant features. "
                     "The model included 85,667,355 parameters, of which, 14,175,774 were trainable. "
                     "We used an Adam optimizer, with inital learning rate and weight decay "
                     "set to 1e-3 and 1e-4, respectively. As in the other models, we used "
                     "Focal Loss as the criterion.")

        st.markdown("<h3>ViT Results</h3>", unsafe_allow_html = True)

        st.write("During the training phase, the learning rate was lowered to 1e-4 and 1e-5 at "
                     "Epoches 10 and 20, respectively. We were able to achieve a weighted F1-Score "
                     ".585 on the validation data. The final test set revealed a weighted F1-Score "
                     "of .59. The full results from the training process and the classification "
                     "report are below.")

        img = Image.open("images/ViT_results.png")
        st.image(img, use_container_width = False)

        img = Image.open("images/classification_report_ViT.png")
        st.image(img, use_container_width = False)

        st.write("We also extracted features for a subset of the images to better "
                     "understand what the Custom ResNet50 model interpreted as important "
                     "features.")


        img = Image.open("images/ViT_gradcam.png")
        st.image(img, use_container_width = False) 



