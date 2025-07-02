# LSR-for-CTI-classification

This repository supports the paper **"LSR: Latent Space Refinement for Cyber Threat Intelligence Classification"**, which proposes an unsupervised text classification method to identify and organize Cyber Threat Intelligence (CTI) text from open-source data.

## 📊 Datasets

Three datasets are used in the study:
### 1. TD Dataset(self-collected and included in this repository)
### 2. BC Dataset (from work Cyberthreat detection from twitter using deep neural networks)
### 3. STC Dataset (from work Corpus and deep learning classifier for collection of cyber threat indicators in twitter stream)

### 1. TD Dataset (Twitter Data)
- The TD dataset consists of tweets grouped into five topics[music, sports, cyber security, movies, health].
- These tweets were collected in 2023 using the Twitter API.
- We provide **tweet IDs only**, in accordance with [Twitter's Developer Policy](https://developer.twitter.com/en/developer-terms/agreement-and-policy), which allows the sharing of up to 50,000 tweet IDs for non-commercial academic research purposes.
- To access the full tweet content, users must **rehydrate** the tweets using their own Twitter API credentials.

### 2. TD Dataset Embeddings
- To enable reproducibility of our results, we provide **SentenceBERT embeddings** generated from the TD dataset using the `sentence-t5-base` model.
- These embeddings are **non-reversible** and do not contain any text, metadata, or identifiers linking back to individuals.
- The embeddings are strictly derived for academic use and are provided for research reproducibility only.

### 3. Anonymity and Privacy
- No personally identifiable information (PII) is included.
- Tweet text, usernames, timestamps, and other metadata are **not** shared.
- Embeddings are not linked to reconstructable content.

## ✅ Ethics Approval
This study was approved by the Queensland University of Technology (QUT) Human Research Ethics Committee under Low-Risk Application No. **LR 2022-6079-10144**.

## 📁 Repository Contents

