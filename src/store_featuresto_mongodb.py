import os
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pymongo

# MongoDB setup
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["audio_files"]
collection = db["features"]

audio_dir = 'audios'

def load_audio_files(audio_directory):
    audio_files = []
    for subdir, dirs, files in os.walk(audio_directory):
        for file in files:
            if file.endswith('.mp3'):
                file_path = os.path.join(subdir, file)
                try:
                    data, sample_rate = librosa.load(file_path, sr=None)  # Keep original sample rate
                    audio_files.append((data, sample_rate))
                    print(f"Successfully loaded file: {file_path}")
                except Exception as e:
                    print(f"Failed to load {file_path}. Error: {e}")
    return audio_files

audio_files = load_audio_files(audio_dir)
print(f"Total files successfully loaded: {len(audio_files)}")

def extract_features(audio_data, sample_rate):
    try:
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13).mean(axis=1)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate).mean(axis=1)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio_data).mean(axis=1)
        features = np.hstack((mfcc, spectral_centroid, zero_crossing_rate))
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

for audio_data, sample_rate in audio_files:
    features = extract_features(audio_data, sample_rate)
    if features is not None:
        document = {"features": features.tolist()}
        collection.insert_one(document)

print("Features successfully stored in MongoDB.")
