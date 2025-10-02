# Sentiment-Analysis-on-Amazon-Product-Reviews
Research_Projects

This project performs sentiment analysis on an Amazon product review dataset. The goal is to build and evaluate various machine learning and deep learning models to classify reviews as either positive (1) or negative (0) based on their text content.

## Table of Contents

  - [Project Overview](https://www.google.com/search?q=%23project-overview)
  - [Dataset](https://www.google.com/search?q=%23dataset)
  - [Methodology](https://www.google.com/search?q=%23methodology)
  - [Models Implemented](https://www.google.com/search?q=%23models-implemented)
  - [Evaluation Results](https://www.google.com/search?q=%23evaluation-results)
  - [Hyperparameter Tuning](https://www.google.com/search?q=%23hyperparameter-tuning)
  - [Dependencies](https://www.google.com/search?q=%23dependencies)
  - [How to Run](https://www.google.com/search?q=%23how-to-run)
  - [Conclusion](https://www.google.com/search?q=%23conclusion)

## Project Overview

The objective of this project is to predict the sentiment of Amazon product reviews. This involves several stages:

1.  **Data Preprocessing**: Cleaning and preparing the raw text data.
2.  **Feature Extraction**: Converting text into a numerical format using TF-IDF.
3.  **Model Training**: Training multiple classification models on the processed data.
4.  **Model Evaluation**: Assessing the performance of each model using standard classification metrics.
5.  **Hyperparameter Tuning**: Optimizing model parameters to improve performance.

## Dataset

The project uses an Amazon product review dataset sourced from a public GitHub repository.

  - **Features**:
      - reviewText: The textual content of the product review.
      - Positive: The sentiment label, where `1` indicates a positive review and `0` indicates a negative review.
  - The dataset is loaded directly into the notebook from the following URL:
    https://raw.githubusercontent.com/rashakil-ds/Public-Datasets/refs/heads/main/amazon.csv
    

## Methodology

1.  **Text Preprocessing**:

      - Converted all text to lowercase.
      - Removed punctuation and special characters.
      - Tokenized the text into individual words.
      - Removed common English stop words.
      - Applied lemmatization to reduce words to their base form (e.g., "running" -\> "run").

2.  **Feature Extraction**:

      - **TF-IDF (Term Frequency-Inverse Document Frequency)**: Used to convert the preprocessed text data into numerical feature vectors for the classical machine learning models. `TfidfVectorizer` was configured with a maximum of 5000 features.
      - **Word Embeddings**: An embedding layer was used for the LSTM model to represent words as dense vectors.

3.  **Train-Test Split**:

      - The dataset was split into a training set (70%) and a testing set (30%) to evaluate model performance on unseen data.

## Models Implemented

Both classical machine learning models and a deep learning model were used for sentiment classification.

#### Statistical Models:

  - **Logistic Regression**
  - **Support Vector Machine (SVM)**
  - **Random Forest Classifier**
  - **Multinomial Naïve Bayes**

#### Neural Network Model:

  - **Long Short-Term Memory (LSTM)**

## Evaluation Results

The models were evaluated on the test set using Accuracy, Precision, Recall, and F1-Score.

| Model                 | Accuracy | Precision | Recall | F1 Score |
| --------------------- | :------: | :-------: | :----: | :------: |
| **SVM** |  0.891   |   0.909   | 0.952  |  0.930   |
| **Logistic Regression** |  0.888   |   0.891   | 0.971  |  0.929   |
| **Random Forest** |  0.868   |   0.871   | 0.969  |  0.917   |
| **Naïve Bayes** |  0.849   |   0.842   | 0.986  |  0.908   |
| **LSTM** |  0.757   |   0.757   | 1.000  |  0.862   |

**Key Observation**: The SVM model achieved the highest F1-score and accuracy, indicating the best balance between precision and recall. The LSTM model performed poorly, classifying all test samples as positive, resulting in a perfect recall but low precision and accuracy. This suggests the model architecture or training process may need further refinement.

## Hyperparameter Tuning

`GridSearchCV` was used to find the optimal hyperparameters for several models.

  - **Logistic Regression**:

      - **Best Parameters**: `{'C': 10, 'solver': 'liblinear'}`
      - **Best Cross-Validation Score**: 88.76%

  - **Naive Bayes**:

      - **Best Parameters**: `{'alpha': 0.1}`
      - **Best Cross-Validation Score**: 86.76%

## Dependencies

This project requires Python 3 and the following libraries:

  - `pandas`
  - `numpy`
  - `nltk`
  - `scikit-learn`
  - `tensorflow`
  - `seaborn`
  - `matplotlib`

You can install them using pip:

```bash
pip install pandas numpy nltk scikit-learn tensorflow seaborn matplotlib
```

## How to Run

1.  **Clone the repository**:

    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

2.  **Install the dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

    *(Note: You may need to create a `requirements.txt` file with the libraries listed above.)*

3.  **Run the Jupyter Notebook**:

    ```bash
    jupyter notebook "Sentiment Analysis on Amazon Product Reviews.ipynb"
    ```

## Conclusion

This project successfully implemented and compared several models for sentiment analysis on Amazon reviews.

  - **Findings**: Classical machine learning models, particularly SVM and Logistic Regression, performed exceptionally well when combined with TF-IDF features. The SVM model provided the best overall performance with an accuracy of 89.1%.
  - **Challenges**: The deep learning (LSTM) model's performance was suboptimal. This could be due to a simple architecture, the need for more data, or the requirement for more extensive hyperparameter tuning (e.g., learning rate, number of layers, dropout).
  - **Key Takeaway**: For this particular text classification task, traditional ML models with robust feature engineering (TF-IDF) proved more effective and computationally efficient than a basic deep learning model.
