# **GTD Terrorism Severity Prediction (Machine Learning & NLP Project)**

This project builds a hybrid **machine learning + NLP classification system** to predict the **severity level** of terrorist attacks using the **Global Terrorism Database (GTD)**.

It includes:

- Custom TF-IDF feature engineering  
- Domain-specific text cleaning  
- Phrase-token engineering  
- One-hot encoded metadata  
- Tuned **Linear SVM** classifier  
- End-to-end training + inference pipeline  
- Multi-model comparison with metrics  

A full NLP + ML + Feature Engineering production-ready project.

---

## Project Highlights

- Processed and cleaned **200K+ GTD records**  
- Created severity labels (Low, Medium, High) from casualty counts  
- Designed optimized TF-IDF (1–2 grams, sublinear TF, 3000 features)  
- Engineered metadata features:
  - One-hot categorical features  
  - Numeric features (success, suicide, coordinates)  
  - Custom severity-related phrase tokens  
- Evaluated multiple ML models:
  - Linear SVM  
  - Logistic Regression  
  - Passive-Aggressive  
  - Naive Bayes  
- Final model achieved:
  - **Accuracy: 0.86**  
  - **Macro F1: 0.77**  

---

## Dataset Used

Source: Global Terrorism Database (GTD)  

Columns used:

- summary  
- region_txt  
- country_txt  
- attacktype1_txt  
- targtype1_txt  
- weaptype1_txt  
- success  
- suicide  
- latitude  
- longitude  
- casualties (converted to severity labels)

Dataset not included; must be downloaded manually.

---

## Machine Learning Pipeline

### **1. Exploratory Data Analysis**
- Missing values  
- Casualty distributions  
- Category label cleanup  
- Severity label creation  

---

### **2. Text Preprocessing**
- Lowercasing  
- Removing punctuation and numeric codes  
- Stopword filtering (NLTK + custom domain words)  
- Removing location/reporting phrases  
- Cleaned text saved as **clean_summary**  

---

### **3. Feature Engineering**

#### **A. One-Hot Encoding (Categorical Features)**

- region_txt  
- country_txt  
- attacktype1_txt  
- targtype1_txt  
- weaptype1_txt  

#### **B. Numerical Features**

- success  
- suicide  
- latitude  
- longitude  

#### **C. Optimized TF-IDF Vectorizer**

- max_features = 3000
- min_df = 2
- ngram_range = (1,2)
- sublinear_tf = True


#### **D. Custom Phrase Tokens**

- token_suicide_attack  
- token_car_bomb  
- token_hostage  
- token_severe  

#### **E. Final Combined Feature Vector**

TF-IDF + One-Hot + Numerical + Custom Tokens  

---

## Models Included

### **1. Linear SVM (Final Model)**  
Best performance across all metrics.

### **2. Logistic Regression**  
Strong baseline.

### **3. Passive-Aggressive Classifier**  
Suitable for large sparse datasets.

### **4. Complement Naive Bayes**  
Text-focused baseline.

### **5. Ridge / SGD Classifier**  
Underperformed.

---

## Model Comparison

| Model | Macro F1 | High F1 | Medium F1 | Accuracy |
|-------|----------|----------|------------|-----------|
| **Linear SVM (Hybrid)** | **0.77** | **0.64** | **0.73** | **0.86** |
| Logistic Regression | 0.72 | 0.63 | 0.67 | 0.83 |
| Passive-Aggressive | 0.67 | 0.52 | 0.62 | 0.77 |
| Complement Naive Bayes | 0.63 | 0.45 | 0.58 | 0.74 |
| Ridge Classifier | 0.54 | 0.42 | 0.40 | 0.66 |
| SGD Classifier | 0.27 | 0.01 | 0.14 | 0.50 |

---

## 📁 Folder Structure

GTD-Severity-Prediction/

│── README.md

│── GTD_Severity_Prediction.ipynb

└── requirements.txt 

---

## Tech Stack

| Component | Technology |
|----------|------------|
| Programming | Python |
| NLP | TF-IDF, Custom Tokens |
| ML Models | Linear SVM, Logistic Regression |
| Vectorization | Scikit-Learn |
| Visualization | Matplotlib, Seaborn |
| Storage | Joblib |

---

## How to Run

Install dependencies:

- pip install -r requirements.txt

Train the model:

- python src/train_model.py

Run predictions:

- python src/predict.py

---

## Accuracy Summary

- Overall Accuracy: **0.86**  
- Macro F1 Score: **0.77**  
- High Severity F1: **0.64**  
- Medium Severity F1: **0.73**  
- Low Severity F1: **0.92**  

---

## Visualizations Included

- Severity distribution  
- TF-IDF vocabulary analysis  
- Model performance comparison  
- Confusion matrices  
- Feature contributions  

---

## Future Enhancements

- Add transformer models (BERT, MiniLM)  
- Implement hierarchical severity scoring  
- Build REST API / Streamlit app  
- Extend to **perpetrator group prediction**  
- Explore BM25 and SCDV feature methods  

---

## Author

**Kanika Singh Rajpoot**  
Data Science Enthusiast

