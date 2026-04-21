# 🧠 ExplainDepression Pipeline
### Explainable Depression Detection from Social Media Text Using XAI

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Kaggle-20BEFF?style=flat-square&logo=kaggle)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-FFD21E?style=flat-square&logo=huggingface)
![Accuracy](https://img.shields.io/badge/Accuracy-96.17%25-brightgreen?style=flat-square)
![F1](https://img.shields.io/badge/F1%20Score-0.9617-brightgreen?style=flat-square)
![AUC](https://img.shields.io/badge/AUC--ROC-0.9937-brightgreen?style=flat-square)

---

## 📌 Overview

**ExplainDepression** is a complete, research-grade pipeline for detecting depression
signals in social media text — with a core focus on *why* the model makes each
decision, not just *what* it decides.

Most depression detection systems output a binary label and stop there.
This pipeline goes further: it produces token-level SHAP explanations for every
prediction, measures whether those explanations are clinically meaningful (DES),
and tests whether the model explains its decisions consistently across different
demographic groups (ECID).

> Built on DistilBERT fine-tuned on 40,770 real Twitter + Reddit posts.
> Trained and evaluated end-to-end on Kaggle GPU T4.

---

## 🏆 Model Performance

| Metric | Score |
|---|---|
| **Test Accuracy** | **96.17%** |
| **F1 Score (Weighted)** | **0.9617** |
| **AUC-ROC** | **0.9937** |
| **Training Time** | **~2 minutes (GPU T4)** |
| **Model** | DistilBERT (fine-tuned, 3 epochs) |
| **Dataset** | 40,770 posts (balanced 50/50) |

---

## 🔬 Three Original Research Contributions

### 1. DES — Depression Explanation Score
A domain-specific metric that measures whether the model's SHAP token
attributions align with DSM-5 clinical indicators of depression.
Rather than raw SHAP values, DES maps high-attribution tokens to validated
clinical categories — hopelessness, anhedonia, social isolation, fatigue,
suicidal ideation, and cognitive distortion.
DES = (triggered_categories / total_categories) × (1 + 0.1 × total_matches)

### 2. ECID — Explanation Consistency Index across Demographics
A fairness metric that detects whether the model explains depression
classifications differently across demographic groups.
A high ECID value signals explanation-level bias — the model is using
ECID(G1, G2) = 1 − cosine_similarity(mean_SHAP_G1, mean_SHAP_G2)

**Real ECID Findings from this pipeline:**

| Clinical Category | SHAP Gap | Status |
|---|---|---|
| Hopelessness | 0.1229 | ⚠️ Bias Signal |
| Social Isolation | 0.0883 | ⚠️ Bias Signal |
| Cognitive Distortion | 0.0893 | ⚠️ Bias Signal |
| Anhedonia | 0.0344 | ✅ Consistent |
| Fatigue | 0.0138 | ✅ Consistent |
| Suicidal Ideation | 0.0348 | ✅ Consistent |
| **Overall ECID** | **0.2063** | ⚠️ Disparity Detected |

### 3. ExplainDepression Pipeline
An end-to-end system that takes raw social media text and produces:
- Depression risk classification (Depressed / Not Depressed)
- SHAP token-level explanation (which words drove the decision)
- DES score (how clinically meaningful the explanation is)
- ECID report (whether the explanation is demographically fair)

---

## 📊 Dataset Summary

| Property | Value |
|---|---|
| Twitter source | stevenhans/depression-and-anxiety-in-twitter-id |
| Reddit source | reihanenamdari/mental-health-corpus |
| Raw combined rows | 34,955 |
| After balancing | 40,770 (20,385 per class) |
| Feature columns | 12 |
| Balancing method | Random Over-Sampling |
| Text cleaning | URL removal, mention stripping, emoji normalisation, lowercase |

---

## 🧬 Clinical Lexicon Categories (DSM-5 Aligned)

| Category | Example Terms | Real DES Score |
|---|---|---|
| Suicidal Ideation | suicide, end my life, wish i was dead | **0.2402** (highest) |
| Social Isolation | alone, lonely, abandoned, disconnected | 0.0726 |
| Hopelessness | hopeless, worthless, no future, give up | 0.0615 |
| Fatigue | exhausted, drained, no energy, lethargic | 0.0438 |
| Cognitive Distortion | always, never, i'm a burden, i ruin everything | 0.0543 |
| Anhedonia | numb, empty, lost interest, can't enjoy | 0.0172 |

---

## 🖼️ Visualisations

### Model Performance Scorecard
![Scorecard](visuals/model_scorecard.png)

### Training History — Loss, F1, AUC-ROC across 3 Epochs
![Training History](visuals/training_history.png)

### Real SHAP Token-Level Explanation Heatmap
![SHAP Heatmap](visuals/shap_token_heatmap_real.png)

### ECID Fairness + DES Clinical Alignment Report
![ECID Chart](visuals/ecid_fairness_chart_real.png)

### Word Cloud — Depression-Related Social Media Terms
![Word Cloud](visuals/wordcloud_depression.png)

---

## 🔬 Research Context

This pipeline addresses three gaps consistently unaddressed in the
mental health NLP literature:

1. No existing depression detection system provides **token-level
clinical explanations** — only binary labels
2. No existing metric measures **explanation quality** against DSM-5
clinical knowledge (this pipeline introduces DES)
3. No existing work evaluates **explanation-level fairness** across
demographic groups (this pipeline introduces ECID and finds real bias)

---

## 🗂️ Data Sources

| Dataset | Platform | Link |
|---|---|---|
| Depression and Anxiety in Twitter (ID) | Kaggle | stevenhans/depression-and-anxiety-in-twitter-id |
| Mental Health Corpus | Kaggle | reihanenamdari/mental-health-corpus |

---

## 🚀 Next Steps

- [ ] Fine-tune full Mental-BERT (110M params) for comparison
- [ ] Expand ECID to real gender and age demographic metadata
- [ ] Benchmark against CLPsych 2015 shared task
- [ ] Build clinical report output format for mental health professionals
- [ ] Submit to ACL / EMNLP workshop on clinical NLP

---

## 📄 License

MIT License

