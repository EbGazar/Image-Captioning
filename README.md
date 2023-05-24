# Medical Image Captioning

Welcome to the Medical Image Captioning Tool repository!

`This repository contains all the necessary documents, design specifications, implementation details and related tools for this Image Captioning Tool that generates natural language captions for Chest X-Rays images!`

You can find the official model implementation in this [Kaggle](https://www.kaggle.com/code/ebrahimelgazar/image-captioning-chest-x-rays) notebook: [Link](https://www.kaggle.com/code/ebrahimelgazar/image-captioning-chest-x-rays)
## Model Architecture


![download](https://github.com/EbGazar/Image-Captioning/assets/62806731/480d8468-358a-4b6f-87fc-2b99bd12e4d1)

The architecture for the model is inspired from "Show and Tell" by Vinyals. The model is built using [Tensorflow](https://www.tensorflow.org/) library.

The overall model used in this project can be categorized as two types: 

1- Image Feature Extraction, using CheXNet. 

2- Caption Generation using LSTM.

CheXNet: the DenseCap model, which is a convolutional neural network
that is pre-trained on the CXR dataset.

`The project also contains code for Attention LSTM layer, although not integrated in the model.`

## Dataset
The model is trained on [Chest X-rays (Indiana University)](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university)

The datasets used for this project are the National Institute of Health Chest X Ray Dataset to train the CNN feature extractor model (CheXNet) and the Chest X-rays (Indiana University) dataset to train the model with the
captions.

`Also it can be trained on any others Medical Dataset`

## Evaluation
- The BLEU score for the test set is 0.64.
- Model Loss: from 12 to 2.0831.

## Requirements
- tensorflow
- keras
- numpy
- h5py
- progressbar2

These requirements can be easily installed by:
  `pip install -r requirements.txt``
  
----------------------------------

## References
[1] Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan. [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/pdf/1411.4555.pdf)

[2]	Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhutdinov, Richard Zemel, Yoshua Bengio. [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/pdf/1502.03044.pdf)`

[3] Kaggle, [Official Model Implementation](https://www.kaggle.com/code/ebrahimelgazar/image-captioning-chest-x-rays)

[4] Official Dataset link, [Chest X-rays (Indiana University)](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university)
