# System Imbalance Forecasting

This repository serves as a supplement to Master's Thesis **Application of transformers for system imbalance prediction in electric power transmission system**. It contains the source codes used in the implementation of the models and all the data (both raw and processed) used for their training and inference. This repository will be further updated after the thesis is defended.

## Abstract

This thesis investigates the application of transformer-based models for predicting system imbalance in the electrical grid. Initially, the study establishes the context of the electricity market and transmission system, highlighting the challenges posed by the integration of variable energy sources and the consequent need for accurate forecasting of system imbalance. It then explores the architecture and capabilities of transformer models, highlighting the use of the attention mechanism. The research meticulously details the preprocessing and use of Belgian electricity market data for model training and evaluation. Furthermore, the implementation of transformer-based models is examined, with an emphasis on specific architectural modifications suitable for time series forecasting. Finally, a comparative analysis is conducted with other machine learning forecasting methods, such as multilayer perceptron and XGBoost.


## Getting Started

Before we begin, it should be noted that Python versions **3.9** and **3.10**, along with their corresponding versions of the `pickle` package, were used. Using other Python versions may lead to unpickling errors.

The repository structure is given as follows:
  - `data`, `TFT_model_data`: Contain data folders.
  - `data_processing`: Includes notebooks for data preprocessing, processing, and other related tasks.
  - `models`: Houses the implemented models.
  - `visualization`: Clear visualization scripts for analysis.

To get started with system imbalance forecasting using this repository, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/obhlivoj/System-Imbalance-Forecasting.git
   cd system-imbalance-forecasting

2. **Install Dependencies:**
   - Using pip:
     ```bash
     pip install -r requirements.txt
     ```
   - Using conda:
     ```bash
     conda install --file requirements.txt
     ```

3. **Explore Model Predictions Interactive Playground:**
    - Execute the `playground.ipynb` Jupyter notebook.
    - It provides an interactive environment for exploring and analyzing the predictions of various forecasting models.
