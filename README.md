# Creating A Music Recommendation Model utilising Apache Spark's MLlib

## Overview
This music recommendation model is a simplified music streaming platform inspired by Spotify. It features an intelligent recommendation engine powered by audio features and machine learning, and delivers music through a responsive Flask web interface.

This project demonstrates key concepts in content-based recommendation, music similarity detection using vector models, and distributed computing with Apache Spark. Audio features such as MFCC, Spectral Centroid, and Zero Crossing Rate are extracted and stored in MongoDB, and similar tracks are retrieved using Locality-Sensitive Hashing (LSH) on Apache Spark.


## Dataset

We use the **Free Music Archive (FMA)** dataset for this project — an open-source collection of audio tracks across multiple genres.

- **Tracks**: 106,574
- **Genres**: 161
- **Duration**: 30s per clip
- **Size**: ~93GB (compressed)

**Download the dataset**: https://github.com/mdeff/fma 
> For local development or limited processing, we recommend using the `fma_small` or `fma_medium` subsets.

## Objective

Build a personalized music recommendation and streaming system that:

- Extracts meaningful audio features.
- Stores features in a scalable database.
- Trains a similarity-based model using Apache Spark.
- Ranks and recommends similar tracks in real-time.
- Streams music through a responsive web interface.


## Technologies Used

- **Python** – Core logic and feature processing
- **Librosa** – Audio feature extraction (MFCC, Spectral Centroid, ZCR)
- **MongoDB** – Audio feature storage
- **Apache Spark** – Distributed model training (LSH)
- **Flask** – Web server and audio player
- **Bootstrap / HTML5** – Frontend styling and responsiveness
- **Apache Kafka** – Real-time recommendations (planned)


## Workflow Overview

### 1. Audio Feature Extraction
Extracted using:
- **MFCC (Mel Frequency Cepstral Coefficients)**
- **Spectral Centroid**
- **Zero Crossing Rate**

Stored as numeric vectors in MongoDB.

### 2. Recommendation Model (LSH)
- Features are loaded into Apache Spark.
- Scaled using `MinMaxScaler`.
- Similar tracks are discovered using **Bucketed Random Projection LSH**.
- `approxSimilarityJoin()` identifies top matches based on Euclidean distance.

### 3. Web Application
- Flask app lists all available tracks from `audios`.
- Bootstrap-styled UI allows user to stream songs.
- **Future:** Real-time feedback and dynamic playlist updates via Apache Kafka.

---

## How to Run

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/Creating-A-Music-Recommendation-Model-utilising-MLlib-of-Apache-Spark.git
cd Creating-A-Music-Recommendation-Model-utilising-MLlib-of-Apache-Spark
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Run Feature Extraction**

```bash
python src/audio_feature_extractor.py
python src/store_featuresto_mongodb.py
```

4.**Train the Recommendation Model**

```bash
python src/tune_model.py
python src/build_model.py
```
5.**Launch the Web App**
```bash
python src/app.py
```
## Output
**MongoDB:** Stores processed feature vectors

**Spark:** Displays pairs of similar tracks with distance metrics

**Web:** Audio player interface for song selection and playback

## Deployment Notes
This project runs well on local systems for small subsets (e.g., fma_small). For full dataset support:

Use cloud-based Spark clusters (e.g., Databricks, EMR)

Scale MongoDB via Atlas or local sharded setup

Integrate Apache Kafka for real-time recommendation








