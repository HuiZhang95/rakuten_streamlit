import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

def image_classification_models():

    st.markdown("<h2>Convolutional Neural Networks</h2>", unsafe_allow_html = True)

    st.write("Convolutional neural networks (CNNs) are used extensively in image "
             "recognition and machine vision models. These deep learning models "
             "extract spatial heirarchies of features through backpropegation. "
             "Models use several types of layers, including convolutional layers, "
             "pooling layers, and fully connected layers in order to process images "
             "for a variety of tasks, including classification (Yamashita et al., 2018). "
             "Below is an example image of CNN architecture.")

    with st.expander("Click to see example CNN architecture"):
        img = Image.open("images/cnn_example.jpg")
        st.image(img, use_container_width = True)

    st.write("For this project, we have use the three pre-trained models as the backbone "
             "for our model architecture, ResNet50 (He et al., 2016), EfficientNetV2 "
             "(Tan & Le, 2021), and ViT (Dosovitskiy et al., 2021).")

    st.markdown("<h2>ResNet</h2>", unsafe_allow_html = True)

    st.write("ResNet represented a breakthrough in CNNs. Prior to ResNet, a major issue "
             "with 'very deep' architectures was that training and validation accuracy "
             "was reduced. This has been described as the degredation problem. This issue "
             "was not related to vanishing/exploding gradients as model architectures "
             "incorporated normalizations which prevented that from occuring. He et al. (2016) "
             "suggested that the degradation problem could be dealt with by incorporating "
             "residual learning. Specifically, the authors introduced 'skip connections' "
             "that explicitly implemented learning of residual features from the previous "
             "convolutional layer.")

    with st.expander("Click to see ResNet50 architecture with 'skip connections'"):
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

    custresnet50 = '''
    class CustomResNet50(nn.Module):
    def __init__(self, base_model):
        super(CustomResNet50, self).__init__()

        self.base_model = base_model
        """
        skip connections are utilized as in the ResNet architecture to
        explicitly facilitate residual learning
        """
        self.skip_connection1 = nn.Conv2d(2048, 2048, kernel_size = 1,
                                          stride = 1, padding = 0)
        """
        additional convolutional layers are structured with normalization,
        pooling, and ReLU activations so their weights can be adjusted to
        the specific needs of our task
        """
        self.Conv1 = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1))

        self.skip_connection2 = nn.Conv2d(2048, 1024, kernel_size = 1,
                                          stride = 1, padding = 0)

        self.Conv2 = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1))

        self.skip_connection3 = nn.Conv2d(1024, 512, kernel_size = 1,
                                          stride = 1, padding = 0)

        self.Conv3 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1))
        """
        averages the the values of input elements to adjust the tensor's
        spatial dimensions
        """
        self.AAPool2d = nn.AdaptiveAvgPool2d((1, 1))
        
        self.Flatten = nn.Flatten(start_dim = 1)
        """
        multiple fully connected layers are connected for classification,
        with both dropout and batch normalization in order to regularize
        the inputs and facilitate convergence
        """
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 27))
    
    def forward(self, x):

        x = self.base_model(x)

        skip_connection1 = self.skip_connection1(x)
        x = self.Conv1(x)
        x = x + skip_connection1
        
        skip_connection2 = self.skip_connection2(x)
        x = self.Conv2(x)
        x = x + skip_connection2
        
        skip_connection3 = self.skip_connection3(x)
        x = self.Conv3(x)
        x = x + skip_connection3

        x = self.AAPool2d(x)
        x = self.Flatten(x)
        x = self.fc(x)

        return x
    
    '''

    with st.expander("Click to see custom architecture added to Resnet50 backbone"):
        st.markdown(f"```python\n{custresnet50}\n```")

    st.markdown("<h3>Custom ResNet50 Results</h3>", unsafe_allow_html = True)

    st.write("The CustomResNet50 new model weights were initialized using kaiming "
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

    with st.expander("Click to see the CustomResNet50 metrics during training"):
        img = Image.open("images/custom_resnet50_results.png")
        st.image(img, use_container_width = False)

    st.write("During the training phase, we were able to achieve a weighted F1-Score "
             "for the validation set of .60. We then evaluated the model on a final "
             "test set and achieved a weighted F1-Score of .59. The full classification "
             "report is below.")

    with st.expander("Click to see full classification report"):
             img = Image.open("images/classification_report_custom_resnet_50.png")
             st.image(img, use_container_width = False)

    st.write("We also extracted features for a subset of the images to better "
             "understand what the Custom ResNet50 model interpreted as important "
             "features.")

    with st.expander("Click to see feature maps for randomly selected images"):
             img = Image.open("images/custom_gradcam.png")
             st.image(img, use_container_width = False)            


    st.markdown("<h2>EfficientNetV2</h2>", unsafe_allow_html = True)

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

    with st.expander("Click to see EfficientNetV2 training time comparisons and fused layers"):
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

    with st.expander("Click to see the trainging and validation metrics."):
        img = Image.open("images/efficientnet_results.png")
        st.image(img, use_container_width = False)

    st.write("During the training phase, we were able to achieve a weighted F1-Score "
             "for the validation set of .60. We then evaluated the model on a final "
             "test set and achieved a weighted F1-Score of .60. The full classification "
             "report is below.")

    with st.expander("Classification results for the EfficientNetV2 model"):
        img = Image.open("images/classification_report_efficientnet.png")
        st.image(img, use_container_width = False)

    st.write("We also extracted features for a subset of the images to better "
             "understand what the Custom ResNet50 model interpreted as important "
             "features.")

    with st.expander("Click to see feature maps for randomly selected images"):
             img = Image.open("images/efficientnet_gradcam.png")
             st.image(img, use_container_width = False) 

    st.markdown("<h2>ViT</h2>", unsafe_allow_html = True)

    st.write("We additionally trained a transformer, ViT (Dosovitskiy et al., 2021), as "
             "part of our project development. Transformers are the standard for training "
             "and developing large language models, but until recently have not been used "
             "with machine vision. To achive this, preset 'patch sizes' of images are sent "
             "through a transformer, where a learnable token 'classification' token is "
             "added to each sequence.")

    with st.expander("Click to see a diagram of an image transformer"):
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

    with st.expander("Click to see the training results for the ViT."):
        img = Image.open("images/ViT_results.png")
        st.image(img, use_container_width = False)

    with st.expander("Click to see the classification report for the ViT."):
        img = Image.open("images/classification_report_ViT.png")
        st.image(img, use_container_width = False)

    st.write("We also extracted features for a subset of the images to better "
             "understand what the Custom ResNet50 model interpreted as important "
             "features.")

    with st.expander("Click to see feature maps for randomly selected images"):
             img = Image.open("images/ViT_gradcam.png")
             st.image(img, use_container_width = False) 



