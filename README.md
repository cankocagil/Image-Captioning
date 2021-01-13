# Image Captioning

## In Brief, ##
This repo is mainly focused on image captioning task using the state of the art techniques in the context of deep learning. Image captioning is the process of generating textual description of an image by using both natural language processing and computer vision applications. Network consists of Convolutional Neural Network (CNN) to encode images into latent space representations followed by Recurrent Neural Network (RNN) to decode feature and word representations and build language models. Specifically, Long-Short Term Memory(LSTM) and Gated Recurrent Unit (GRU) are used as a RNN model with attention mechanism and teacher forcer algorithm. To realize that, transfer learning applications such as AlexNet, VGG-Net, ResNet, DenseNet and SquezeeNet are used as a convolutional encoder and Global Vector for Word Representation(GloVe) is used for word embedding. Flickr dataset is used for both training and testing. Various data augmentation techniques are implemented to boost the model performance. The model is compiled by Adam optimizer with scheduled learning rates. Masked Cross Entropy loss is used for criterion for the models. Finally, beam and greedy search algorithms are implemented to get the best image-to-caption translation.

- - - -


 * Keywords
    * Convolutional Language Model for Image Captioning
    * Deep Learning for Vision & Language Translation Models
    * Transfer Learning
    * Data Augmentation
    * Parallel Distibuted Processing
    * Attention model and Teacher Forcer Algorithm
    * AlexNet, VGG-Net, ResNet, DenseNet and SquezeeNet
    * Long-Short Term Memory(LSTM) and Gated Recurrent Unit (GRU)
    * Global Vector for Word Representation(GloVe) 
    * Beam and Greedy search
    * BLEU Scores and METEOR

- - - -

Here are some samples from our Vision & Language Models:
    
- ![Caption1](https://user-images.githubusercontent.com/53329652/104514695-b7e39200-5602-11eb-8352-175d7fc6219f.png) 
- ![Caption3](https://user-images.githubusercontent.com/53329652/104514705-bb771900-5602-11eb-8a54-e016de65449c.png) 


Please see ImageCaptionPaper for documentation of this repo and all generated captions. 
