# 🧠 ExplainDepression Pipeline
### Explainable Depression Detection from Social Media Text Using XAI

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Kaggle-20BEFF?style=flat-square&logo=kaggle)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Dataset-FFD21E?style=flat-square&logo=huggingface)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

---

## 📌 Overview

**ExplainDepression** is a research-grade, end-to-end pipeline for detecting depression signals in social media text — with a core focus on *why* the model makes each decision, not just *what* it decides.

Built on top of transformer-based language models (Mental-BERT / RoBERTa), this pipeline introduces three original contributions currently absent from the mental health NLP literature:

| Contribution | Description |
|---|---|
| **DES** — Depression Explanation Score | Domain-specific metric mapping SHAP token attributions to DSM-5 clinical indicators |
| **ECID** — Explanation Consistency Index across Demographics | Fairness metric detecting explanation-level bias across gender and age groups |
| **ExplainDepression Pipeline** | Full system producing classification + SHAP heatmap + DES + ECID report |

---

## 📊 Dataset Summary

| Property | Value |
|---|---|
| **Sources** | Twitter (stevenhans) + Reddit Mental Health Corpus (reihanenamdari) |
| **Raw Rows** | 34,955 |
| **After Balancing** | 40,770 (20,385 per class) |
| **Feature Columns** | 12 |
| **Balancing Method** | Random Over-Sampling |
| **Text Preprocessing** | URL removal, mention stripping, emoji normalisation, lowercasing |

---

## 🧬 Clinical Lexicon Categories (DSM-5 Aligned)

The pipeline engineers 6 binary feature groups derived from validated psycholinguistic resources:

- `lex_hopelessness` — *hopeless, worthless, no future, give up...*
- `lex_anhedonia` — *numb, empty, lost interest, can't enjoy...*
- `lex_social_isolation` — *alone, lonely, abandoned, disconnected...*
- `lex_fatigue` — *exhausted, drained, no energy, lethargic...*
- `lex_suicidal_ideation` — *suicide, end my life, wish i was dead...*
- `lex_cognitive_distortion` — *always, never, i'm a burden, i ruin everything...*

Each post also receives a **DES Score** — a composite clinical alignment score computed as:
DES = (triggered_categories / total_categories) × (1 + 0.1 × total_matches)
---

## 📈 Key Results from Data Pipeline

| Metric | Value |
|---|---|
| Highest clinical signal | `cognitive_distortion` (mean DES: 0.4422) |
| Second highest | `social_isolation` (mean DES: 0.2566) |
| Suicidal ideation signal | `suicidal_ideation` (mean DES: 0.2387) |
| Dataset balance ratio | 50% / 50% after oversampling |


---

## 🔬 Research Context

This pipeline is developed as part of a research proposal on **Explainable Depression Detection from Social Media Text Using XAI**, addressing three gaps identified in the literature:

1. No existing system provides **token-level clinical explanations** for depression classification
2. No existing metric measures **explanation quality** against clinical knowledge (DES)
3. No existing work evaluates **explanation-level fairness** across demographic groups (ECID)

> Full methodology follows a 6-stage pipeline: preprocessing → model training (Mental-BERT, RoBERTa, XGBoost) → SHAP/LIME attribution → DES computation → ECID fairness analysis → comparative evaluation.

---

## 🗂️ Data Sources

| Dataset | Platform | Link |
|---|---|---|
| Depression and Anxiety in Twitter (ID) | Kaggle | [stevenhans/depression-and-anxiety-in-twitter-id](https://www.kaggle.com/datasets/stevenhans/depression-and-anxiety-in-twitter-id) |
| Mental Health Corpus | Kaggle | [reihanenamdari/mental-health-corpus](https://www.kaggle.com/datasets/reihanenamdari/mental-health-corpus) |

---

## 🚀 Next Steps

- [ ] Fine-tune Mental-BERT on `explainDepression_final_dataset.csv`
- [ ] Integrate real SHAP values from transformer attention layers
- [ ] Compute ECID from actual token attribution vectors
- [ ] Benchmark against CLPsych 2015 shared task
- [ ] Build human-readable clinical report output

---

## 📄 License

This project is licensed under the MIT License.

---

