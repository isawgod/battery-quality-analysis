# 🔋 Battery Quality Analysis & Defect Detection  
*Using NASA PCoE Li-ion Battery Dataset (B0005, B0006, B0007, B0018)*

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-brightgreen?logo=streamlit)](https://battery-quality-analysis-wnyf23tyn3kfm4ds8n2rph.streamlit.app/))

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

📈 *Example plots from notebook:*  
- Capacity fade curves
- Correlation heatmap  
- Classification confusion matrix  
- RUL prediction vs true  

---

## 8. 📑 Insights & Summary of Findings

Through the exploratory analysis and modeling, several important insights emerged about lithium-ion battery degradation:

- **Capacity fade is non-linear**  
  - All cells show relatively stable capacity in early cycles.  
  - Degradation accelerates rapidly once capacity drops below ~85% of the initial value.  
  - This “knee” behavior makes forecasting RUL challenging with simple linear models.

- **Cycles-to-failure vary significantly by cell**  
  - Battery **B0005** failed much earlier (~170 cycles) compared to **B0006** and **B0007** (>400 cycles).  
  - This demonstrates manufacturing and material variability between cells, even under controlled test conditions.

- **Key predictors of failure**  
  - **Minimum voltage** during discharge: consistently decreases as cells approach failure.  
  - **Fade rate** (capacity drop per cycle): sharp negative spikes often precede rapid degradation.  
  - **Mean temperature**: higher internal and ambient temperatures correlate with faster degradation.  
  - Impedance (where available) tends to rise as capacity fades, suggesting it is a strong future feature.

- **Threshold-based defect labeling works well**  
  - Defining defective cycles as those with <70% of initial capacity aligns with NASA’s reliability guidelines.  
  - Clear separation between “healthy” and “defective” groups enables supervised classification.

- **Modeling outcomes**  
  - **Random Forest** achieved the best defect classification results (higher F1 and ROC-AUC than Logistic Regression).  
  - **Gradient Boosting Regressor** predicted Remaining Useful Life more accurately than simple Linear Regression, especially near mid-life.  
  - However, all models tend to slightly **underestimate RUL mid-life** and **overestimate near end-of-life** — a known challenge due to non-linear fade.

- **Operational insight**  
  - Early monitoring of fade rate and voltage drops could enable predictive maintenance before catastrophic failure.  
  - Cells exposed to higher ambient or operating temperatures degrade faster, reinforcing the importance of **thermal management**.

---

## 🚀 How to Run

### 1. Clone Repository
```bash
git clone https://github.com/isawgod/battery-quality-analysis.git
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

## 11. 🔑 Key Takeaways

- **Degradation is non-linear**: Cells fade slowly at first, then rapidly after a “knee point” around ~85% capacity.  
- **Variability between cells**: Batteries of the same type can fail at very different cycle counts, underscoring manufacturing and usage variability.  
- **Defect detection**: Random Forests outperform linear models by capturing complex, non-linear relationships.  
- **RUL estimation**: Gradient Boosting provided the most accurate RUL predictions, though non-linear fade remains challenging.  
- **Practical impact**: These methods can improve quality control, reduce warranty costs, and enable predictive maintenance in real-world deployments.  
- **Future improvements**: Incorporating richer time-series features (capacity slopes, voltage hysteresis) and impedance data will further boost model reliability.

---

## 📝 Author
**Przemyslaw Teodor Rydz**  
Data Scientist | Battery Analytics 

📫 Connect with me on [LinkedIn](https://www.linkedin.com/in/przemyslaw-rydz-a2a55633b)  

## 🙏 Special Thanks
Special thanks to my lovely girlfriend, I love You V.
