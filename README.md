## 🧩 Week 1 – Design Phase Summary

### 🧠 Problem Statement
The world faces increasing environmental challenges due to improper waste disposal and inefficient segregation. Urbanization and changing lifestyles have made manual waste sorting unsafe, inefficient, and unsustainable.  
To address this issue, the project aims to design an **AI-based Garbage Classification System** that can automatically classify waste using deep learning techniques.

### 💡 Solution Approach
An intelligent and automated system is proposed using **Convolutional Neural Networks (CNN)** for accurate image-based waste classification.  
The CNN model will be trained to recognize different waste categories (plastic, paper, metal, etc.) and support sustainable recycling through automation.

### 🗂️ Dataset Information
- **Dataset Name:** Garbage Classification Dataset  
- **Source:** Kaggle ([Garbage Classification V2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2))  
- **Description:** Contains images of garbage items categorized into 10 classes such as plastic, metal, cardboard, glass, and paper.  
- **Purpose:** Designed for machine learning and computer vision projects focused on recycling and waste management.

### 🧱 Design Activities
- Identified dataset and analyzed data distribution.  
- Designed CNN structure with multiple convolution and pooling layers.  
- Selected `TensorFlow/Keras` as the framework and `Google Colab` for GPU training.  
- Planned preprocessing techniques (resizing, normalization, and train-test split).  

**Outcome:**  
Week 1 successfully completed the system design phase, finalized architecture, dataset source, and planned the model training process.


## 💻 Week 2 – Implementation Phase Summary

### ⚙️ Implementation Overview
During Week 2, the designed CNN model was implemented and trained using the **Garbage Classification Dataset** on Google Colab (with GPU acceleration).

### 🧩 Implementation Steps
1. Imported the Kaggle dataset using the Kaggle API.  
2. Preprocessed images using `ImageDataGenerator` for scaling and augmentation.  
3. Built and compiled a CNN model using TensorFlow/Keras with:
   - Conv2D and MaxPooling2D layers  
   - Flatten and Dense layers  
   - Dropout for overfitting control  
4. Trained the model for multiple epochs and monitored accuracy/loss graphs.  
5. Tested sample images and verified successful predictions.

### 📊 Results
- **Training Accuracy:** 80.7%  
- **Validation Accuracy:** 69.9%  
- Model saved as: `waste_classifier_model.h5`  
- Verified prediction with test images using `plt.imshow()`  

### 🧾 Files Added to GitHub
- `Garbage_Classification_Week2_Implementation.ipynb` – Implementation notebook  
- `model_link.txt` – Google Drive link for the trained model (since file >25MB)  
- `sample_prediction.png` and `accuracy_loss_graph.png` (optional visuals)

**Outcome:**  
Week 2 completed the CNN model implementation phase, successfully achieving high accuracy and demonstrating automatic waste classification through deep learning.
