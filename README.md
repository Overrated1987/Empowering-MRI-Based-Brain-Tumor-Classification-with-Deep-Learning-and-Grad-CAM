
# **Empowering MRI-Based Brain Tumor Classification with Deep Learning and Grad-CAM**

Brain tumor classification poses a significant challenge in computer-aided diagnosis (CAD) for medical applications. Machine learning algorithms hold promise for assisting radiologists in reliably detecting tumors, potentially reducing the need for invasive procedures. Despite this, key issues remain, such as selecting the optimal deep learning architecture and ensuring accurate assessment of model outputs. This study addresses these challenges by proposing an advanced deep learning-based system for classifying four types of brain tumors—Glioma, Meningioma, Pituitary tumors, and non-tumor images—using a comprehensive MRI dataset.

The proposed system integrates various deep learning models through transfer learning, tailored specifically for the Brain Tumor MRI Dataset. It employs extensive data preprocessing and augmentation techniques to enhance model performance. The dataset, sourced from Kaggle, includes 7,023 MRI scans, making it one of the largest datasets available for brain tumor classification. Experimental results indicate that the system significantly improves classification accuracy, achieving approaching 99\% with ResNet-50, Xception, and InceptionV3 models.

A thorough comparative analysis was conducted, benchmarking the developed models against state-of-the-art methods in the literature. To address issues of transparency and interpretability in deep learning models, Grad-CAM was utilized to visualize decision-making processes for tumor classification in MRI scans. Additionally, a user-friendly Brain Tumor Detection System was developed using Streamlit, enhancing user interaction and accessibility in medical diagnostics. This system demonstrates practical applicability in real-world settings and provides a valuable tool for clinicians.

## **Problem Statement**

Brain tumor is the accumulation or mass growth of abnormal cells in the brain. There are basically two types of tumors, malignant and benign. Malignant tumors can be life-threatening based on the location and rate of growth. Hence timely intervention and accurate detection is of paramount importance when it comes to brain tumors. This project focusses on classifying 3 types of brain tumors based on its loaction from normal cases i.e no tumor using Convolutional Neural Network.

## Proposed Framework

![Structure of the Proposed Framework for Brain Tumor Classification](Framework7.17%(1)-cropped.pdf-cropped.pdf)

The proposed framework for brain tumor classification is illustrated in the figure above. It represents a high-level overview of the process using MRI scans to classify brain tumors. The methodology includes several key steps:

1. **Dataset Acquisition**: The Brain Tumor MRI Dataset, which includes images of meningioma, glioma, and pituitary tumors, was sourced from publicly available repositories.

2. **Image Preprocessing**: To improve data quality, comprehensive image preprocessing techniques were applied. This step was crucial for enhancing the reliability of subsequent analyses.

3. **Data Partitioning and Augmentation**: The dataset was randomly divided into training, validation, and testing sets. Image augmentation techniques were exclusively applied to the training set to increase its diversity and robustness.

4. **Model Training**: Six pre-trained models—VGG19, ResNet50, Xception, MobileNetV2, InceptionV3, and NASNetLarge—were fine-tuned for the classification task. These models were initialized with pre-trained weights and adjusted to fit the specific classes (meningioma, pituitary, and glioma).

5. **Performance Evaluation**: The framework's effectiveness was evaluated using various metrics, including accuracy, specificity, sensitivity, F1-score, and confusion matrix.

6. **Interpretability**: To improve transparency, Grad-CAM was employed to visualize the decision pathways of the models. This technique provided valuable insights into how the models arrived at their predictions based on MRI scans.

## **Dataset**

The dataset utilized for this model is sourced from the Brain Tumor MRI Dataset available on Kaggle. It consists of MRI images categorized into different types of brain tumors and non-tumor cases.

### Categories

The dataset is divided into the following categories:

![Categories of the Brain Tumor MRI Dataset](dataset-cropped.pdf)

### Dataset Details and Distribution

The dataset is partitioned into training, validation, and testing sets as outlined below:

| **Category**       | **Training** | **Validation** | **Testing** |
|--------------------|--------------|----------------|-------------|
| Glioma tumor       | 1,060        | 261            | 300         |
| Meningioma tumor   | 1,072        | 267            | 306         |
| Pituitary tumor    | 1,158        | 299            | 300         |
| No tumor           | 1,279        | 316            | 405         |
| **Total**          | **4,569**    | **1,143**      | **1,311**   |

This distribution ensures a comprehensive training process, effective validation, and robust testing for the model's performance evaluation.


### **Image Preprocessing**
Image preprocessing is applied to all the images in the dataset
1. Cropping the image : removes the unwanted background noise. Thus helping the algorithm to focus completely on the features of interest

<p align="center">
  <img src="https://github.com/saras56/Brain_Tumor_Detection_Multiclass/assets/115695360/5ce30227-f438-4fc2-bd25-8ee51f0c828b" alt="Description of the image">
</p>
<p align="center">
  Images after Cropping
</p>

2.	Noise Removal : Bilateral filter is used for noise removal. It smooths the image while preserving edges and fine details. Bilateral filter considers both the spatial distance and intensity similarity between pixels when smoothing the image. Hence suitable for processing MRI images acquired with different imaging protocols and parameters.
3.	Applying colormap : Applying a colormap can improve the interpretability of MRI images by enhancing the contrast between different tissues or structures
4.	Resize : Resizing the image for standardizing the input size of images to be fed into a machine learning model

<p align="center">
  <img src="https://github.com/saras56/Brain_Tumor_Detection_Multiclass/assets/115695360/0fc6573c-2c9e-43b6-afcc-cd61a2c7172c" alt="Description of the image">
</p>
<p align="center">
  Images after preprocessing
</p>


### **Splitting the data into train, test and validation**
Here the train data is split into train and validation sets. The test data is completely unseen. There are 2912 train images, 729 validation images an 893 test images.


### **Image Augmentation using Image Data Generator**
Medical imaging datasets, including MRI images, are often limited in size due to factors such as data collection constraints, privacy concerns, or rarity of certain conditions. Image augmentation allows to artificially increase the size of the dataset by generating variations of existing images. Augmentation can help prevent the model from memorizing specific patterns or features in the training data that may not generalize well to unseen data, thus leading to a more robust and generalizable model.

### **Model Training**

Resnet-50 is used for training the brain tumor dataset. ResNet-50’s increased depth allows it to capture more intricate patterns and features in the data, which can be beneficial for detecting complex structures in brain tumor images. By transfer learning, ResNet-50’s pre-trained weights from ImageNet are leveraged to bootstrap training on the brain tumor classification task. 

## **Results**
The following results have been achieved with Resnet-50 model for detection of Glioma, Meningioma, Pituitary and Normal patients from Brain MRI images.

- Test Accuracy      : 97%
- f1-score (glioma)  : 97%
- f1-score (meningioma) : 96%
- f1-score (pituitary) : 96%
- f1-score (no_tumorl) : 100%


**Confusion matrix**

<p align="center">
  <img src="https://github.com/saras56/Brain_Tumor_Detection_Multiclass/assets/115695360/98b38811-b4ef-4ad6-b3a1-d6a75b25219a">
</p>

**Sample predictions**

Predicted label(True label)

<p align="center">
  <img src="https://github.com/saras56/Brain_Tumor_Detection_Multiclass/assets/115695360/9dc134eb-2753-46ea-b081-2a9793c55e3d">
</p>

## **Streamlit App**

<p align="center">
  <img src="https://github.com/saras56/Brain_Tumor_Detection_Multiclass/assets/115695360/fe226ebe-d080-420b-83b7-c81e7ed37df7" alt="Description of the image">
</p>
<p align="center">
  Prediction for Pituitary tumor
</p>

## **Future work**
- Include more image preprocessing steps so as to extract intricate details correctly 
- Increase the number of samples in the dataset
