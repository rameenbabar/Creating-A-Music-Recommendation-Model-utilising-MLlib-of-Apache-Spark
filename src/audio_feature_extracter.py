import os
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Define the root directory containing nested folders of audio files
audio_dir = 'audios'

# Function to load audio files, skipping any that fail to load
def load_audio_files(audio_directory):
    audio_files = []
    for subdir, dirs, files in os.walk(audio_directory):
        for file in files:
            if file.endswith('.mp3'):  # Check for mp3 files
                file_path = os.path.join(subdir, file)
                try:
                    data, sample_rate = librosa.load(file_path, sr=None)  # Load at original sample rate
                    audio_files.append((data, sample_rate))
                    print(f"Successfully loaded file: {file_path}")
                except Exception as e:
                    print(f"Failed to load {file_path}. Skipping this file: {e}")
    return audio_files

# Load audio files
audio_files = load_audio_files(audio_dir)
print(f"Total files successfully loaded: {len(audio_files)}")

# Feature extraction functions
def extract_features(audio_data, sample_rate):
    try:
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13).mean(axis=1)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate).mean(axis=1)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio_data).mean(axis=1)
        features = np.hstack((mfcc, spectral_centroid, zero_crossing_rate))
        print(f"Extracted features: {np.array2string(features, formatter={'float_kind':lambda x: f'{x:0.2f}'})}")

        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None  # Return None if there's an error

# Function to preprocess features
def preprocess_features(features):
    try:
        # Standardize features
        scaler = StandardScaler()
        standardized_features = scaler.fit_transform(features)
        print(f"Standardized features: {standardized_features[0]}")  # Print the first sample's features
        
        # Reduce dimensions
        pca = PCA(n_components=0.95)  # Adjust n_components according to your preference
        reduced_features = pca.fit_transform(standardized_features)
        print(f"Features after PCA: {reduced_features[0]}")  # Print the first sample's features after PCA
        
        return reduced_features
    except Exception as e:
        print(f"Error in preprocessing features: {e}")
        return None

# Process all audio files and extract features
all_features = []
for audio_data, sample_rate in audio_files:
    features = extract_features(audio_data, sample_rate)
    if features is not None:
        all_features.append(features)

# Only create DataFrame if there are features to process
if all_features:
    all_features_array = np.array(all_features)
    preprocessed_features = preprocess_features(all_features_array)
    features_df = pd.DataFrame(preprocessed_features)
    print("Features prepared for ML model training:", features_df.shape)
else:
    print("No features extracted. Check the audio files and extraction process.")
