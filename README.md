
# Relative Embedding for Periocular and Face Recognition: Conditional Multimodal Biometrics

This repository provides the implementation and experiments for the project **"Relative Embedding for Periocular and Face Recognition: Conditional Multimodal Biometrics"**, conducted at Yonsei University.

---

## Motivation
The COVID-19 pandemic has reduced the accuracy of conventional face recognition because masks cover important facial areas. To address this, we focus on the **periocular region** (the area around the eyes) as a reliable biometric trait. We also propose a **Conditional Multimodal Biometrics (CMB)** framework that combines periocular and face recognition for stronger identification and verification.

---

## Methodology
### End-to-End Deep Face Recognition
- **Face Detection**: Locate faces in images  
- **Face Alignment**: Normalize faces using key landmarks  
- **Face Representation**: Extract features with a MobileFaceNet backbone

### Conditional Multimodal Biometrics (CMB)
- Combines periocular and face embeddings  
- Introduces a **CMB Loss** that:
  - Pulls embeddings from the same subject closer (even across modalities)  
  - Pushes embeddings from different subjects further apart  
- Supports four recognition tasks:
  - Periocular ↔ Periocular (p2p)  
  - Face ↔ Face (f2f)  
  - Periocular ↔ Face (p2f)  
  - Face ↔ Periocular (f2p)

### Similarity Metrics
- **Cosine Similarity** (main measure)  
- **KL Divergence** (tested for cross-modal similarity)  

---

## Software Implementation
- Identification and verification pipelines using extracted embeddings  
- **Metrics**:
  - CMC (Cumulative Match Curve) for identification  
  - EER (Equal Error Rate), ROC, and AUC for verification  
- KL divergence optimized with broadcasting for faster computation  

---

## Datasets
Experiments were performed on six datasets:
- **Ethnic**  
- **PubFig**  
- **FaceScrub**  
- **IMDB-Wiki**  
- **AR** (including blur, occlusion, and scarf conditions)  
- **YTF (YouTube Faces)**

---

## Results
- **Periocular-to-Periocular**: CMC up to ~97%, EER as low as 2.8%  
- **Face-to-Face**: CMC up to ~98%, EER ~2.4%  
- **Cross-modal (p2f, f2p)**: CMC up to ~97%, EER around 5%  
- **YTF dataset**: Lower performance due to low resolution  

Cosine similarity generally outperformed KL divergence.

---

## Suggested Repository Structure
