# Time Cross Adaptive Self Attention( TCSA ) based Imputation model

 Time-Cross Adaptive Self-Attention (TCSA) model  for multivariate Time Series Imputation Model for Sparse  and Irregular Data

 ## model Architecture 
 
![Model archi Diagram drawio](https://github.com/user-attachments/assets/00851210-1a15-403d-92bd-2d68bd184f28)



# ⏳ Time-Cross Adaptive Self-Attention (TCSA) for Multivariate Time Series Imputation

This repository presents a novel deep learning architecture—**Time-Cross Adaptive Self-Attention (TCSA)**—designed for imputing missing values in sparse and irregular multivariate time series data. The model leverages temporal and cross-variable dependencies using adaptive attention mechanisms, making it ideal for real-world datasets with non-uniform sampling and high missingness.

## 📌 Project Overview

Multivariate time series data often suffer from missing values due to sensor failures, irregular sampling, or transmission errors. Traditional imputation methods fail to capture temporal dynamics and inter-variable relationships. TCSA addresses this by:

- Modeling both **temporal** and **cross-variable** dependencies
- Using **adaptive self-attention** to weigh relevant time steps and features
- Handling **sparse and irregular** data without interpolation

## 🧠 Model Architecture

The TCSA model consists of the following components:

### 🔹 Temporal Encoder
- Captures intra-variable temporal patterns
- Uses self-attention over time steps
- Learns dynamic temporal weights

### 🔹 Cross-Variable Encoder
- Captures inter-variable correlations
- Applies attention across feature dimensions
- Enables adaptive feature fusion

### 🔹 Fusion Layer
- Combines temporal and cross-variable embeddings
- Applies residual connections and layer normalization

### 🔹 Imputation Head
- Predicts missing values using fused representations
- Optimized with masked loss functions to focus on missing entries

## 🚀 Technologies Used

- **PyTorch**: Core deep learning framework
- **NumPy & Pandas**: Data manipulation and preprocessing
- **Matplotlib**: Visualization of imputation performance
- **Jupyter Notebook**: Interactive experimentation
- **Python Scripts**: Modular training and evaluation

## 📁 Repository Structure

```text
├── DataSet_file/                   # Raw and processed time series data
├── Modeling/                       # TCSA model implementation
├── dataset_creation_script/       # Scripts for generating synthetic or real datasets
├── config.py                       # Hyperparameter configuration
├── plot.py                         # Visualization utilities
├── random.ipynb                    # Exploratory analysis and testing
├── test.ipynb                      # Evaluation notebook
├── requirement.txt                 # Python dependencies
├── LICENSE                         # GPL-3.0 license
├── README.md                       # Project documentation

