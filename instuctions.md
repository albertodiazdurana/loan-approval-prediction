# Project: Loan Approval Prediction - Mock Interview Prep
Domain: Binary Classification (Financial Services)

## Document References
This document is part of an Integrated_System File → Refer to `0_Integrated_System_Guide-START-HERE.md` for complete guide

## Framework Documents
- **PM Guidelines** (Project Knowledge): Project planning structure and templates
- **Collaboration Methodology** (Project Knowledge): Execution workflow and standards
- **Project Reference** (Project Knowledge): `ProjectReference_Documentation.md`

---

## Project Planning Context

### Scope
- **Purpose**: Prepare baseline notebook for 50-minute interview; build working binary classifier to predict loan approval
- **Resources**: Time: 1 hour total (50 min active), Solo project, No budget constraint
- **Success Criteria**:
  - Quantitative: Working baseline model with performance metrics
  - Qualitative: Clean, interview-ready notebook demonstrating DS workflow
  - Technical: Reproducible, well-documented, follows best practices

### Data & Dependencies
- **Primary dataset**: `loans_modified.csv` (~600 rows, 13 columns)
- **Data dictionary**: `data_dictionary.txt` (column definitions)
- **Target variable**: `loan_status` (1=approved, 0=rejected)
- **Features**: Gender, married, dependents, education, self_employed, incomes, loan details, credit_history, property_area
- **Data quality**: Missing values observed in multiple columns

### Stakeholders & Governance
- **Primary**: Interview Panel (information needs: DS process, technical competence, communication)
- **Communication**: Notebook presentation during interview
- **Governance**: N/A (mock interview prep)

---

## Execution Context

### Timeline & Phases
- **Duration**: 1 hour prep session
- **Deliverable**: 1 notebook (EDA + baseline models)
- **Interview scope**: Build upon prep work with advanced techniques

### Phases (Compressed for Interview Prep)
**Phase 1 - Data Loading & EDA (15 min)**
- Load data, understand structure
- Missing value analysis
- Target distribution
- Feature distributions and correlations

**Phase 2 - Preprocessing & Feature Engineering (15 min)**
- Handle missing values
- Encode categorical variables
- Create train/test split
- Feature scaling (if needed)

**Phase 3 - Baseline Models (15 min)**
- Logistic Regression (baseline)
- Decision Tree
- Random Forest
- Compare performance (accuracy, precision, recall, F1)

**Phase 4 - Documentation (5 min)**
- Add markdown cells explaining workflow
- Document key findings
- Prepare talking points for interview

### Deliverables
- [x] 1 Notebook: Complete EDA + 3 baseline models
- [x] Model comparison table
- [x] Key insights documented in markdown
- [x] Leave room for hyperparameter tuning during interview

---

## Domain Adaptations

### Key Techniques (Binary Classification)
- Class imbalance handling (check target distribution)
- Feature encoding (ordinal vs one-hot for categorical variables)
- Credit history as strong predictor (domain knowledge)
- Income ratios (applicant + coapplicant)
- Model interpretability (important for financial decisions)

### Known Challenges
- **Missing values**: Strategic imputation (median for numeric, mode for categorical)
- **Class imbalance**: Check distribution; prepare SMOTE/class weights if needed
- **Feature engineering**: Income-to-loan ratio, total income
- **Interview time constraint**: Keep preprocessing simple, focus on model performance

---

## Advanced Practices
Select from Methodology's "Advanced Complexity" section:

- [ ] Experiment Tracking (not needed for prep)
- [ ] Hypothesis Management (informal only)
- [x] Performance Baseline (3 algorithms: LogReg, DT, RF)
- [ ] Ethics & Bias Review (acknowledge in interview if asked)
- [ ] Testing Strategy (not production)
- [ ] Data Versioning (single dataset)
- [ ] Risk Management (not applicable)
- [ ] Technical Debt Register (not applicable)
- [ ] Scalability Planning (small dataset)
- [ ] Literature Review (not needed)

---

## Communication & Style

### Artifact Generation
- Ask clarifying questions before generating
- Confirm understanding of requirements
- Be concise in responses
- Progressive execution: test cell-by-cell, provide output for each cell
- Each output becomes reference for work progress

### Notebook Style
- Clear section headers with markdown
- Inline comments for complex logic
- Display key statistics and visualizations
- Professional formatting for interview presentation

### File Naming Convention
- Format: `wYY_dXX_PHASE_description.ipynb`
- Example: `w01_d01_EDA_baseline_model.ipynb`
- No emojis, professional tone

### Token Monitoring
- Alert at ~180K tokens (95% of 190K practical limit)
- Generate session summary if approaching limit
- Provide context for new chat continuation

---

## Quick Reference

**Data Location**: `loans_modified.csv`  
**Data Dictionary**: `data_dictionary.txt`  
**Target Variable**: `loan_status` (1/0)  
**Algorithms**: Logistic Regression, Decision Tree, Random Forest  
**Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix  
**Time Budget**: 50 minutes total (EDA: 15m, Preprocessing: 15m, Models: 15m, Documentation: 5m)