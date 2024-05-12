from pyspark.sql import SparkSession
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col, udf, monotonically_increasing_id
from pyspark.ml.linalg import VectorUDT
from pyspark.ml.feature import BucketedRandomProjectionLSH

# Initialize Spark Session with MongoDB integration
spark = SparkSession.builder \
    .appName("Music Recommendation System") \
    .config("spark.mongodb.input.uri", "mongodb://localhost:27017/audio_files.features") \
    .getOrCreate()

# Load data from MongoDB
df = spark.read.format("mongo").load()

vector_udf = udf(lambda x: Vectors.dense(x), VectorUDT())
df = df.withColumn("features_vector", vector_udf(df.features))

# Normalize features
scaler = MinMaxScaler(inputCol="features_vector", outputCol="scaledFeatures")
model = scaler.fit(df)
df = model.transform(df)

# Adding unique ID for each track
df = df.withColumn("id", monotonically_increasing_id())

# Setup LSH model for feature vectors
bucket_length = 1.0 
num_hash_tables = 3  # More tables increase the probability of hash collisions
brp = BucketedRandomProjectionLSH(inputCol="scaledFeatures", outputCol="hashes", bucketLength=bucket_length, numHashTables=num_hash_tables)
model = brp.fit(df)

# Self-joining to find similar tracks
transformed_df = model.transform(df)
threshold = 1.5  # Adjust this threshold to capture more or fewer similarities
similarity_df = model.approxSimilarityJoin(transformed_df, transformed_df, threshold, distCol="EuclideanDistance") \
                     .filter("datasetA.id != datasetB.id") \
                     .select(col("datasetA.id").alias("id1"),
                             col("datasetB.id").alias("id2"),
                             col("EuclideanDistance"))

# Show top similar pairs
similarity_df.orderBy("EuclideanDistance", ascending=True).show()

# Stop Spark session
spark.stop()

