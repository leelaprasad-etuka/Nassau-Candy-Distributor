# Factory Reallocation and Shipping Optimization Recommendation System

Decision support and logistics optimization platform for Nassau Candy Distributor.

---

## 1. Project Overview

Nassau Candy Distributor operates 5 manufacturing facilities and serves 59 destination jurisdictions across the United States and Canada (49 US states including Washington D.C., and 10 Canadian provinces). Historically, each of the 15 confectionery products has been manufactured exclusively at a single historical facility, leading to cross-country transit routes, extended delivery lead times, and increased freight expenditure.

This system provides:
1. Data cleaning, financial integrity verification, and IQR outlier rejection.
2. Geospatial modeling using haversine distance bridging manufacturing plants to state centroids.
3. KMeans route corridor clustering evaluated with silhouette analysis.
4. Predictive modeling (Linear Regression, Random Forest, Gradient Boosting) evaluating lead times.
5. Counterfactual scenario simulation evaluating 17,700 production permutations.
6. Multi-criteria optimization ranking reallocations based on speed gain, profit impact, and risk penalty.
7. An interactive Streamlit decision dashboard (`app.py`) featuring 4 operational modules.
8. Rigorous technical documentation (`reports/research_paper.md`) and non-technical briefing (`reports/executive_summary.md`).

---

## 2. Project Directory Structure

```
.
├── app.py                              # Primary Streamlit application entrypoint
├── requirements.txt                    # Pinned package dependencies for deployment
├── README.md                           # Documentation and instructions
├── data/
│   ├── Nassau_Candy_Distributor.csv    # Raw transactional dataset
│   ├── cleaned_nassau_candy.csv        # Cleaned dataset (IQR filtered)
│   └── processed_nassau_candy.csv      # Feature-engineered dataset
├── src/
│   ├── config.py                       # Configuration parameters and paths
│   ├── data_prep.py                    # Stage 1: Ingestion, cleaning, financial checks, IQR
│   ├── features.py                     # Stage 2: Spatial coordinates, distance, margins
│   ├── geo.py                          # Centroid lookup and vectorised haversine formula
│   ├── clustering.py                   # Stage 3: Route aggregation and KMeans clustering
│   ├── models.py                       # Stage 4: 5-fold CV, regression benchmarks, artifact saving
│   ├── simulation.py                   # Stage 5: Grid simulation and tree uncertainty engine
│   └── recommend.py                    # Stage 6: Multi-criteria scoring and operational KPIs
├── models/
│   ├── best_lead_time_model.joblib     # Serialized Random Forest model
│   ├── feature_scaler.joblib           # StandardScaler fitted on training set
│   ├── feature_encoders.joblib         # OneHotEncoder fitted on training set
│   ├── model_evaluation_metrics.joblib # 5-fold CV and test evaluation metrics
│   ├── route_clustering_model.joblib   # KMeans model and cluster profiles
│   └── precomputed_scenarios.csv       # 17,700 precomputed counterfactual allocations
├── notebooks/
│   └── eda.ipynb                       # Comprehensive EDA and visual evidence notebook
└── reports/
    ├── research_paper.md               # Full academic and technical research paper
    └── executive_summary.md            # One-page executive summary for stakeholders
```

---

## 3. Installation and Setup

### Prerequisites
- Python 3.10 or higher (Python 3.11 recommended)

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 4. Pipeline Execution (Stage-by-Stage Verification)

Execute each stage in sequential order:

```bash
# Stage 1: Data audit, financial validation, and IQR outlier removal
python -m src.data_prep

# Stage 2: Feature engineering and spatial distance computation
python -m src.features

# Stage 3: EDA and route clustering (optimal k=5 via silhouette score)
python -m src.clustering

# Stage 4: Train models, run 5-fold CV, evaluate test set, and save models
python -m src.models

# Stage 5: Run counterfactual scenario simulation (17,700 evaluations)
python -m src.simulation

# Stage 6: Compute multi-criteria rankings and operational KPIs
python -m src.recommend
```

---

## 5. Launching the Streamlit Application

Launch the interactive dashboard locally or when deploying:

```bash
streamlit run app.py
```

### Application Modules:
1. **Factory Optimization Simulator:** Select any SKU and inspect predicted lead time, adjusted profit impact, and confidence across all 5 manufacturing facilities in tables and comparative charts.
2. **What-If Scenario Analysis:** Compare current vs proposed manufacturing locations on an interactive North American geospatial map with overlay corridor lines.
3. **Recommendation Dashboard:** Review ranked portfolio proposals, monitor the 4 operational KPI metric cards, and download recommendations as a CSV file.
4. **Risk and Impact Panel:** Monitor low-confidence flags, excessive freight increase warnings, predictive model performance tables, and data limitation notices.

---

## 6. Deployment Guide (Streamlit Community Cloud)

When deploying to Streamlit Community Cloud:
1. Push this repository to GitHub.
2. Connect your GitHub repository in Streamlit Cloud.
3. Set **Main file path** to: `app.py`.
4. Deploy! Streamlit Cloud will automatically detect `requirements.txt` in the root folder and launch `app.py`.
