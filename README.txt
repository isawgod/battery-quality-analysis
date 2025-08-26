# 🔋 Battery Quality Analysis & Defect Detection  
*Using NASA PCoE Li-ion Battery Dataset (B0005, B0006, B0007, B0018)*

## 📌 Project Overview
Lithium-ion batteries are critical in electric vehicles, aerospace, and energy storage systems.  
However, they degrade over time, leading to:
- Reduced usable capacity
- Increased internal resistance
- Potential safety risks

This project demonstrates an **end-to-end data science workflow**:
1. **Exploratory Data Analysis (EDA)** on raw NASA battery test data.
2. **Defect Detection**: Classify cycles as *healthy* or *defective* based on capacity.
3. **Remaining Useful Life (RUL) Prediction**: Estimate how many cycles remain until a cell reaches its end-of-life.
4. **Deployment**: Provide a **Streamlit app** for interactive predictions.

---

## 📂 Dataset
- Source: [NASA Ames Prognostics Center of Excellence (PCoE)](https://www.nasa.gov/content/prognostics-center-of-excellence-datasets)
- Batteries analyzed: **B0005, B0006, B0007, B0018**
- Data format: MATLAB `.mat` files containing:
  - Voltage, current, temperature, impedance measurements
  - Cycle-level capacity (discharge capacity per cycle)
- Failure definition: **Capacity < 70% of initial capacity**

---

## 🧪 Methods

### 1. Exploratory Data Analysis
- Capacity fade curves for all four batteries
- Time-series plots of voltage, temperature, and current
- Impedance trends (where available)
- Correlation analysis between features

### 2. Feature Engineering
- **Fade rate** (∆ capacity per cycle)
- **Min voltage** during discharge
- **Mean temperature** during discharge
- **Mean current**
- **Discharge duration**
- **Ambient temperature**

### 3. Machine Learning Models
- **Defect Detection (Classification)**
  - Logistic Regression (baseline)
  - Random Forest Classifier (best)
- **RUL Prediction (Regression)**
  - Linear Regression (baseline)
  - Gradient Boosting Regressor (best)

### 4. Deployment
- Streamlit app (`app.py`) for:
  - Uploading cycle-level feature CSVs
  - Predicting defect probability
  - Estimating Remaining Useful Life (RUL)

---

## 📊 Results

### Defect Classification
- Random Forest achieved higher F1 and ROC-AUC vs Logistic Regression
- Top predictors:
  - Minimum voltage
  - Fade rate
  - Mean temperature

### RUL Prediction
- Gradient Boosting Regressor achieved lowest MAE/RMSE
- Predicted RUL curves closely follow true degradation patterns
- Residuals show underestimation mid-life, overestimation near end-of-life

📈 *Example plots from notebook (replace with actual screenshots):*  
- Capacity fade curves  
- Correlation heatmap  
- Classification confusion matrix  
- RUL prediction vs true  

---

## 🚀 How to Run

### 1. Clone Repository
```bash
git clone https://github.com/<your-username>/battery-quality-analysis.git
cd battery-quality-analysis
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Run Notebook
```bash
jupyter lab
# open notebooks/eda_modeling.ipynb
```

### 4. Run Streamlit App
```bash
streamlit run app.py
```

Upload a CSV with columns:  
```
fade_rate, duration, min_voltage, mean_current, mean_temp, ambient_temp
```

Try with the sample file provided:  
```
demo_cycles.csv
```

---

## 📦 Repository Structure
```
battery-quality-analysis/
├─ data/                 # .mat files (not included; download from NASA)
├─ notebooks/
│  └─ eda_modeling.ipynb # Full EDA + Modeling pipeline
├─ app.py                # Streamlit app
├─ clf_rf.joblib         # Trained Random Forest classifier
├─ rul_gbr.joblib        # Trained Gradient Boosting regressor
├─ demo_cycles.csv       # Demo input CSV for Streamlit
├─ requirements.txt
└─ README.md
```

---

## 🔮 Next Steps
- Engineer rolling-window features (capacity slope over last N cycles)
- Integrate impedance features more systematically
- Experiment with advanced models (XGBoost, LightGBM, LSTMs)
- Deploy as a full **web dashboard** with real-time monitoring

---

## 📝 Author
Przemyslaw Rydz 
Data Scientist | Battery Analytics | Predictive Maintenance  

📫 Connect with me on [LinkedIn](www.linkedin.com/in/przemyslaw-rydz-a2a55633b)  

---

### 🙏 Special Thanks
Special thanks to my lovely gf I love You V.