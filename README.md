# Loan Approval Prediction - Binary Classification

**Domain:** Financial Services | **Task:** Binary Classification | **Purpose:** Interview Preparation

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Project Overview

End-to-end machine learning project predicting loan approval status based on applicant demographics, income, credit history, and loan details. Built as a 50-minute interview preparation exercise demonstrating full data science workflow from EDA to optimized model deployment.

**Repository:** [github.com/albertodiazdurana/loan-approval-prediction](https://github.com/albertodiazdurana/loan-approval-prediction)

---

## Key Features

- **Binary Classification:** Predict loan approval (1) or rejection (0)
- **Dataset:** ~600 loan applications with 13 features
- **Advanced Models:** Baseline comparison + optimized Random Forest with GridSearchCV
- **Class Imbalance Handling:** SMOTE implementation
- **Model Interpretability:** SHAP analysis for global and individual prediction explanations
- **Comprehensive Evaluation:** ROC curves, feature importance, cross-validation, confusion matrices
- **Interview-Ready:** Clean notebook + 100-question Q&A guide

---

## Project Structure

```
loan-approval-prediction/
├── data/
│   ├── raw/                      # Original datasets
│   │   ├── loans_modified.csv    # Loan applications data
│   │   └── data_dictionary.txt   # Feature definitions
│   └── processed/                # Cleaned/encoded data
├── notebooks/
│   └── w01_d01_EDA_baseline_models.ipynb  # Main prep notebook
├── models/                       # Saved model artifacts
├── outputs/
│   └── figures/                  # Visualizations
├── docs/
│   ├── plans/                    # Project planning documents
│   │   └── LoanApproval_Week1_Plan.md
│   └── guides/
│       ├── ProjectReference_Documentation.md
│       └── Loan_Approval_Prediction_QA.md  # interview questions
├── setup_domain_extensions.py   # Install ML packages
├── requirements_base.txt         # Base dependencies
└── README.md                     # This file
```

---

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/albertodiazdurana/loan-approval-prediction.git
cd loan-approval-prediction
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate or .venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
# Install base packages
pip install -r requirements_base.txt

# Install domain-specific ML packages
python setup_domain_extensions.py
```

### 4. Launch Jupyter
```bash
jupyter notebook notebooks/
```

### 5. Run Main Notebook
Open `w01_d01_EDA_baseline_models.ipynb` and execute cells sequentially.

---

## Dataset Description

**Source:** `/data/raw/loans_modified.csv` (~600 rows × 13 columns)

| Feature              | Type        | Description                             |
| -------------------- | ----------- | --------------------------------------- |
| `loan_id`            | Categorical | Unique loan identifier                  |
| `gender`             | Binary      | Male / Female                           |
| `married`            | Binary      | Yes / No                                |
| `dependents`         | Ordinal     | 0, 1, 2, 3+                             |
| `education`          | Binary      | Graduate / Not Graduate                 |
| `self_employed`      | Binary      | Yes / No                                |
| `applicant_income`   | Numeric     | Applicant's income                      |
| `coapplicant_income` | Numeric     | Co-applicant's income                   |
| `loan_amount`        | Numeric     | Loan amount (thousands)                 |
| `loan_amount_term`   | Numeric     | Loan term (months)                      |
| `credit_history`     | Binary      | Credit history meets guidelines (1 / 0) |
| `property_area`      | Categorical | Urban / Semiurban / Rural               |
| **`loan_status`**    | **Binary**  | **Target: 1 (approved), 0 (rejected)**  |

**Data Quality Issues:**
- Missing values in: gender, married, dependents, self_employed, loan_amount, loan_amount_term, credit_history
- Categorical encoding required for non-numeric features

---

## Methodology

### Phase 1: Exploratory Data Analysis (15 min)
- Load and inspect dataset structure
- Analyze target distribution (check class imbalance)
- Identify missing values
- Visualize feature distributions and correlations

### Phase 2: Data Preprocessing 
- Handle missing values (median for numeric, mode for categorical)
- Encode categorical variables (label encoding + one-hot)
- Feature engineering: `total_income`, `income_to_loan_ratio`
- Train/test split (80/20)

### Phase 3: Baseline Modeling 
- Train 3 baseline models: Logistic Regression, Decision Tree, Random Forest
- Evaluate with accuracy, precision, recall, F1-score
- Compare performance with confusion matrices
- Identify best-performing model

### Phase 4: Model Optimization
- Apply SMOTE for class imbalance
- GridSearchCV hyperparameter tuning (108 combinations)
- 5-fold cross-validation
- ROC curve and AUC analysis
- Feature importance visualization

### Phase 5: Documentation
- Add professional markdown headers
- Document key findings and model comparison
- Create comprehensive Q&A guide for interview

---

## Model Performance

**See notebook for detailed results:** `w01_d01_EDA_baseline_models.ipynb`

**Best Model:** Random Forest (optimized with GridSearchCV)
- **Training Accuracy:** ~99% (after hyperparameter tuning)
- **Test Accuracy:** ~80% (balanced performance)
- **Cross-Validation:** 5-fold CV shows consistent performance
- **ROC-AUC Score:** Strong discrimination capability

**Key Performance Factors:**
- SMOTE applied for class imbalance handling
- 108 hyperparameter combinations tested via GridSearchCV
- Feature importance analysis reveals credit_history as top predictor
- Confusion matrix analysis quantifies false positive/negative rates

---

## Key Insights

1. **Credit History:** Strongest predictor (confirmed through feature importance and SHAP analysis - 0.1109 mean |SHAP|)
2. **Engineered Features:** income_to_loan_ratio ranks 2nd in SHAP importance (0.0382), validating feature engineering
3. **Income Ratios:** Total income and income-to-loan ratio provide context for loan affordability
4. **Class Imbalance:** SMOTE tested but decreased performance; original distribution optimal
5. **Model Complexity:** Random Forest outperforms simpler models after optimization
6. **Generalization:** Cross-validation confirms model stability across data splits
7. **Interpretability:** SHAP analysis provides transparent explanations for individual predictions

---

## Business Value

### ROI Analysis
- **Cost of False Negative:** Missed revenue opportunity (declined viable applicant)
- **Cost of False Positive:** Default risk (approved risky applicant)
- **Model Impact:** Optimized threshold based on business cost function

### Stakeholder Communication
- **Loan Officers:** Automated pre-screening reduces manual review time
- **Risk Management:** Quantified default probability for portfolio management
- **Executives:** Clear metrics on approval accuracy and revenue impact

### Production Considerations
- **Model Interpretability:** Feature importance enables transparent decisions
- **Bias Audit:** Gender/demographic fairness analysis required before deployment
- **Regulatory Compliance:** Document decision criteria for loan approval

---

## Interview Enhancements Completed

- [x] Hyperparameter tuning with GridSearchCV (108 combinations)
- [x] Class imbalance analysis with SMOTE
- [x] Feature importance visualization
- [x] ROC curve and AUC score analysis
- [x] 5-fold cross-validation
- [x] Confusion matrix analysis
- [x] SHAP analysis for model interpretability
- [x] Business value quantification
- [x] 95-question interview Q&A guide

---

## Future Enhancements (Production)

- [ ] Gradient Boosting models (XGBoost, LightGBM)
- [ ] Threshold optimization based on business cost functions
- [ ] Ensemble methods (stacking, voting classifier)
- [ ] Polynomial features and interaction terms
- [ ] Advanced feature selection (RFE, recursive elimination)
- [ ] Fairness audit (demographic bias testing)
- [ ] A/B testing framework for production deployment
- [ ] Real-time monitoring dashboard
- [ ] Automated retraining pipeline

---

## Interview Preparation

### Q&A Guide: 100 Technical Questions

**File:** [Loan_Approval_Prediction_QA.md](docs/guides/Loan_Approval_Prediction_QA.md)

Comprehensive interview preparation covering:
- **20 sections** matching notebook structure
- **5 questions per section** (100 total)
- Technical depth (algorithms, metrics, trade-offs)
- Business awareness (ROI, risks, stakeholder communication)
- Production readiness (deployment, monitoring, maintenance)

**Question types:**
- Why decisions were made
- What-if scenarios
- Trade-off analysis
- Stakeholder translation
- Production concerns

**Key sections:**
- Model training & evaluation (Sections 10-12)
- Optimization attempts (Sections 14-15)
- Feature importance (Section 16)
- SHAP interpretability (Section 20)
- Cross-validation (Section 18)
- Production readiness (Section 19)

---

## Technologies Used

**Core:**
- Python 3.9+
- Jupyter Notebook

**Data Processing:**
- pandas
- numpy

**Visualization:**
- matplotlib
- seaborn

**Machine Learning:**
- scikit-learn (LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GridSearchCV)
- imbalanced-learn (SMOTE)
- shap (model interpretability)

---

## Documentation

- **Project Plan:** [LoanApproval_Week1_Plan.md](docs/plans/LoanApproval_Week1_Plan.md)
- **Project Reference:** [ProjectReference_Documentation.md](docs/ProjectReference_Documentation.md)
- **Data Dictionary:** [data_dictionary.txt](data/raw/data_dictionary.txt)
- **Interview Q&A:** [Loan_Approval_Prediction_QA.md](docs/Loan_Approval_Prediction_QA.md)

---

## License

MIT License - See [LICENSE](LICENSE) for details

---

## Author

**Alberto Diaz Durana**  
[GitHub](https://github.com/albertodiazdurana) | [LinkedIn](https://www.linkedin.com/in/albertodiazdurana/)
---

**Status:** Completed | **Last Updated:** November 2025