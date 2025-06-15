# Multimodal Recommender System for Personalized Product Suggestions

<br>
<div align="center">
<img src="demo_image/demo_2Tower.avif" width="300"/>
</div>
<br>

This project designs and implements a **deep learning-based recommender system** using a **two-tower neural architecture**, capable of delivering scalable and personalized product suggestions. It combines **structured data**, **textual product descriptions**, and **product images** to improve recommendation performance.

The model is trained on the [H&M Personalized Fashion Recommendations dataset](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) from Kaggle.

<br>

## Key Features

- **Two-Tower Neural Network**: Separate encoders for users and items, enabling efficient candidate retrieval at scale  
- **Multimodal Input**: Combines structured features with image and text embeddings  
- **Transformer Encoder Integration**: A BERT-like module is integrated into the system trained from scratch to distill richer information from grouped modalities  

<br>

## Data Modalities

The system integrates:

- **User Information**:
  - User ID, age, active status, fashion news frequency, etc.

- **Product Information**:
  - Product type, textual description, product image (processed into embeddings), etc.

- **Transaction Logs**:
  - User–item purchases with metadata (price, timestamp, sales channel)

<br>

## Major Files

- `Recommender_2Tower__main_.py`  
  Main script for running the model

- `Recommender_2Tower__NNmodel_.py`  
  The neural network model codes that contains the Two-Tower architecture and Transformer-based encoders

- `Recommender_2Tower__NNDataset_.py`  
  PyTorch Dataset class for loading and batching multimodal data

- `Recommender_2Tower__train_.py`  
  Training script with early stopping 
