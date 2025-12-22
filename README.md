# 🏦 Credit Risk Lab  
Framework Clean Architecture pour le Scoring Crédit & l’Approbation de Prêts

## 📌 Description
**Credit Risk Lab** est un framework modulaire, extensible et production-ready dédié au développement de modèles de scoring crédit (loan approval, loan risk, creditworthiness).

Ce projet applique une **Clean Architecture strictement respectée**, permettant :

- une séparation claire entre domain, application et infrastructure  
- une meilleure testabilité  
- une maintenabilité optimale  
- la possibilité d'ajouter ou remplacer facilement des modèles (XGBoost, CatBoost, LightGBM, LogReg…)  
- l'intégration de pipelines de feature engineering structurés  
- un repository de modèles (filesystem ou S3)  
- un monitoring (MLflow, drift detection, PSI, KS-test)  
- une exposition via CLI ou API

Le framework est conçu pour être utilisé en entreprise dans des environnements MLOps réels.

---

# 📊 Loan Approval Classification Dataset  
**Source des données :** [Kaggle - Loan Approval Classification](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data)

## Dataset Overview
Ce dataset (synthetic) contient **45 000 instances** et **14 variables** liées au risque crédit et à l’approbation de prêts.  
Il a été enrichi et étendu via **SMOTENC** pour augmenter les instances tout en respectant les structures catégorielles.

### 📌 Variables principales
- **person_age** — âge  
- **person_gender** — genre  
- **person_education** — niveau d'éducation  
- **person_income** — revenu annuel  
- **person_emp_exp** — années d’expérience professionnelle  
- **person_home_ownership** — statut de logement  
- **loan_amnt** — montant de prêt demandé  
- **loan_intent** — motif du prêt  
- **loan_int_rate** — taux d’intérêt  
- **loan_percent_income** — % du revenu consacré au prêt  
- **cb_person_cred_hist_length** — ancienneté de l’historique crédit  
- **credit_score** — score crédit  
- **previous_loan_defaults_on_file** — défauts antérieurs  
- **loan_status** (target) — 1 = approuvé, 0 = refusé

---

# 🧱 Architecture du Projet (Clean Architecture)

```

credit-risk-lab/
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
├── .pre-commit-config.yaml
├── tests/
│   ├── test_domain/
│   ├── test_application/
│   ├── test_infrastructure/
│   └── test_interfaces/
└── src/
└── credit_risk_lab/
├── domain/
│   ├── entities.py
│   ├── value_objects.py
│   ├── services.py
│   └── ports.py
│
├── application/
│   ├── dto.py
│   └── use_cases/
│       ├── train_model_uc.py
│       ├── score_application_uc.py
│       ├── evaluate_model_uc.py
│       └── bascule_model_uc.py
│
├── infrastructure/
│   ├── data_sources/
│   ├── features/
│   ├── ml_backends/
│   ├── model_store/
│   └── monitoring/
│
├── interfaces/
│   ├── cli.py
│   └── api.py
│
├── config/
└── shared/

````

---

# 🧩 Modules Principaux

## 🧠 Domain (Business Logic Pure)
- Entités métier (Applicant, LoanRequest, CreditScore…)  
- Règles métier  
- Ports (interfaces abstraites)

## 🚀 Application (Use Cases)
- Entraînement du modèle  
- Scoring d'application de crédit  
- Évaluation & comparaison modèles  
- Bascule modèle (champion/challenger)

## 🏗 Infrastructure
- Repositories datasets  
- Feature engineering pipelines (WOE, binning, encoders)  
- Backends ML (XGBoost, CatBoost, LightGBM, Logistic Regression)  
- Stockage de modèles (Filesystem, S3)  
- Monitoring (MLflow, Drift detection PSI/KS)

## 🌐 Interfaces
- CLI (training, scoring, evaluation)  
- API (FastAPI, optionnel)

---

# 📦 Installation

```bash
git clone https://github.com/<yourname>/credit-risk-lab.git
cd credit-risk-lab

poetry install
````

Créer un fichier `.env` si nécessaire :

```
MLFLOW_TRACKING_URI=http://localhost:5000
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

---

# ⚙️ Exemple d’utilisation

## 🎯 1) Entraîner un modèle

```bash
poetry run python -m credit_risk_lab.interfaces.cli train \
    --config configs/training_config.yaml
```

## 🧪 2) Évaluer un modèle

```bash
poetry run python -m credit_risk_lab.interfaces.cli evaluate
```

## 🔍 3) Scorer une nouvelle application de crédit

```bash
poetry run python -m credit_risk_lab.interfaces.cli score \
    --input app.json
```

---

# 📈 MLflow Tracking

Tous les modèles, métriques et artefacts sont suivis via MLflow :

* AUC
* KS
* Recall / Precision
* Confusion matrix
* Feature importance
* Versionning des modèles
* Champion / Challenger

---

# 🧪 Tests Unitaires

```bash
pytest -q
```

Tests structurés par couches :

* `test_domain`
* `test_application`
* `test_infrastructure`
* `test_interfaces`

---

# 🛠 Outils Dev

* **black** (formatage)
* **isort** (imports)
* **mypy** (typage statique)
* **pytest** (tests)
* **pre-commit** (qualité code CI/CD)

Installation :

```bash
pre-commit install
```

---

# 🤝 Contributions

Les PR sont les bienvenues.
Respecte l’architecture clean et la convention des ports / adapters.

---

# 📜 License

MIT License.

---

# 🧑‍💼 Auteur

**Manda Surel**
Machine Learning Engineer — Credit Risk & MLOps

```
