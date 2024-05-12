# Creating a Music Recommendation Model utilising Apache Spark's MLlib

# Introduction
Discovering new music that resonates with your tastes and preferences has never been easier, thanks to cutting-edge music recommendation systems. These systems leverage advanced machine learning algorithms to delve into the intricacies of user behavior and listening habits. By analyzing factors like frequently played tracks, created playlists, and followed artists, these algorithms generate personalized recommendations tailored specifically to each individual's unique musical palette. Drawing from collaborative filtering techniques, content-based filtering, and natural language processing, these systems continuously refine their suggestions, ensuring a diverse and engaging musical journey for users.

# Implementation

In the initial phase of developing our music recommendation model, we focused on extracting essential features from the vast collection of audio tracks. Leveraging Python, we employed a variety of signal processing techniques, including Mel-Frequency Cepstral Coefficients (MFCCs), spectral centroid, and zero-crossing rate. These methods enabled us to transform raw audio data into numerical and vector formats, capturing crucial aspects of each track's acoustic characteristics. 

## Feature Extraction

### Mel-Frequency Cepstral Coefficients(MFCCs)
Mel-Frequency Cepstral Coefficients (MFCCs) are an important tool in signal processing and audio analysis, commonly used for tasks such as speech and music recognition. They provide a compact representation of the spectral envelope of an audio signal. Specifically, MFCCs capture the short-term power spectrum of a sound by converting the frequency domain of the signal into a set of coefficients. This process involves mapping the audio signal onto a mel-scale, which is a perceptual scale of pitches based on human hearing. The conversion to the mel-scale allows MFCCs to better mimic the human auditory system's sensitivity to different frequencies. In our implementation, MFCCs are computed using the librosa.feature.mfcc() function, which returns the MFCCs of the audio data. The parameter n_mfcc specifies the number of MFCCs to compute, allowing flexibility in the level of detail captured in the representation. Finally, the mean of MFCCs is calculated along the axis to obtain a single MFCC feature vector for each audio file, which can then be used as input for further analysis or modeling tasks.

### Spectral Centroid
Spectral centroid is a fundamental feature in audio signal processing that serves as a measure of the distribution of spectral energy in a sound. It represents the "center of mass" or the weighted mean of the power spectrum of an audio signal, indicating where the "center of gravity" of the spectrum is located. Higher spectral centroid values typically correspond to sounds with higher frequencies, reflecting a shift towards the higher end of the frequency spectrum. In our code, spectral centroid is computed using the librosa.feature.spectral_centroid() function, which extracts this characteristic from the audio data. Subsequently, the mean of spectral centroid values is calculated along the axis to derive a single spectral centroid feature vector for each audio file.

### Zero-Crossing Rate
Zero-crossing rate is a significant metric in audio signal analysis, quantifying the rate at which a signal changes its polarity, or crosses the zero axis. It offers valuable insights into the frequency of changes or rapid fluctuations in the amplitude of the signal, reflecting the presence of high-frequency components or sudden transitions in the audio waveform. Higher zero-crossing rates often indicate higher frequencies or more rapid changes in the signal, making it a useful indicator of signal dynamics. In this implementation, zero-crossing rate is computed using the librosa.feature.zero_crossing_rate() function, which extracts this characteristic from the audio data. Subsequently, the mean of zero-crossing rate values is calculated along the axis to derive a single zero-crossing rate feature vector for each audio file.

### Storing Feature Vectors on MongoDB
To address the dataset's considerable size and ensure scalability and accessibility, we opted for MongoDB as our storage solution. Its seamless scalability and accessibility capabilities make it well-suited for our requirements. Following the feature extraction phase, we effortlessly loaded the audio features into a MongoDB collection. This approach facilitated efficient storage and retrieval of feature data, enabling further analysis as needed.

## Music Recommendation Model
In our Music Recommendation System, we seamlessly integrate MongoDB and Apache Spark to develop an efficient and scalable recommendation model.  Leveraging Spark's capabilities, we employ its DataFrame API to seamlessly interface with MongoDB, enabling us to load the audio features directly into Spark for further processing. We utilize advanced techniques such as Locality Sensitive Hashing (LSH) within Apache Spark to develop an efficient recommendation model. LSH is a powerful algorithm for approximate nearest neighbor search, enabling us to efficiently identify similar items within our dataset. By leveraging LSH within Apache Spark, our recommendation model efficiently identifies similar tracks in our dataset, paving the way for personalized music recommendations tailored to each user's preferences.

## Locality Sensitive Hashing(LSH) within Apache Spark
To implement LSH, we start by normalizing the audio features using MinMaxScaler, ensuring that each feature is within a consistent range for accurate comparison. Next, we employ the BucketedRandomProjectionLSH algorithm, a variant of LSH, to hash the normalized feature vectors into buckets based on their similarity. By tuning parameters such as bucket length and the number of hash tables, we control the trade-off between computational efficiency and accuracy in similarity detection.

With LSH in place, we perform a self-join operation on the dataset to find similar tracks. This involves comparing each track's feature vector with every other track's vector to identify potential matches based on their proximity in the hashed space. The resulting similarity DataFrame captures pairs of tracks deemed similar according to a specified threshold, typically based on Euclidean distance.

### How does Euclidean Distance measure similarity?
Euclidean distance is commonly used in many machine learning algorithms, particularly when dealing with vector spaces, because of its intuitive geometrical interpretation. It's effective for problems where the magnitudes of the attribute vectors are important. Here’s why it's suitable for our music recommendation system:

**Intuitive Interpretation:** The closer the vectors are, the smaller the Euclidean distance, indicating higher similarity.

**Effectiveness with MFCCs and Other Audio Features:** For audio tracks, features like MFCCs capture aspects related to pitch and timbre, where differences can often be effectively quantified using linear distances.

**Symmetry:** Euclidean distance is symmetric, meaning the distance from track A to track B is the same as from B to A.

### Hyperparameter Tuning
Hyperparameter tuning in machine learning involves finding the set of optimal parameters for a model that are not learned directly from the data during training. These parameters, known as hyperparameters, govern the training process itself and can significantly affect the performance of the model. In the context of your music recommendation system using the **BucketedRandomProjectionLSH** in PySpark, hyperparameter tuning involves adjusting parameters such as **bucketLength and numHashTables**. These are critical in determining how the hashing for the **Locality-Sensitive Hashing (LSH)** algorithm is performed, which in turn affects the ability to find similar items efficiently.

### How Hyperparameter tuning makes the model better?

**BucketLength:** This parameter defines the width of the buckets into which data points are hashed. Smaller bucket lengths can lead to finer partitions, which may increase the precision of finding true neighbors but might miss some near ones. Larger bucket lengths might hash more distinct points into the same buckets, increasing recall but also false positives. Optimal bucket length ensures that similar items are hashed to the same bucket with higher probability while keeping the dissimilar ones apart, balancing the trade-offs between precision and recall.

**NumHashTables:** This parameter specifies the number of hash tables used in LSH. Increasing the number of hash tables can increase the chance of finding true nearest neighbors as each hash table provides another chance of hashing close neighbors into the same bucket. While more hash tables can improve recall by reducing the chance of missing true similar pairs, it also increases computational resources and query time, creating a trade-off between accuracy and performance.

