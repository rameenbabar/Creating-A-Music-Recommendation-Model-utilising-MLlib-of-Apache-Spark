from pyspark.sql import SparkSession
from pyspark.ml.feature import MinMaxScaler, VectorAssembler, BucketedRandomProjectionLSH
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import col, udf, monotonically_increasing_id

def evaluate_model(model, data):
    transformed_data = model.transform(data)
    similarity_df = model.approxSimilarityJoin(transformed_data, transformed_data, threshold=0.5, distCol="EuclideanDistance") \
                          .filter("datasetA.id != datasetB.id") \
                          .select(col("EuclideanDistance"))
    return similarity_df.agg({"EuclideanDistance": "avg"}).collect()[0][0]

# Initialize Spark Session with MongoDB integration
spark = SparkSession.builder \
    .appName("Music Recommendation System") \
    .config("spark.mongodb.input.uri", "mongodb://localhost:27017/audio_files.features") \
    .getOrCreate()

# Load data from MongoDB
df = spark.read.format("mongo").load()

# Convert features into vector
vector_udf = udf(lambda x: Vectors.dense(x), VectorUDT())
df = df.withColumn("features_vector", vector_udf(df.features))

# Normalize features
scaler = MinMaxScaler(inputCol="features_vector", outputCol="scaledFeatures")
df = scaler.fit(df).transform(df)

# Adding unique ID for each track
df = df.withColumn("id", monotonically_increasing_id())

# Setup LSH model for feature vectors
brp = BucketedRandomProjectionLSH(inputCol="scaledFeatures", outputCol="hashes")

# Manually grid search for best parameters
best_score = float("inf")
best_params = {}
for bucketLength in [1.0, 2.0, 4.0]:
    for numHashTables in [1, 3, 5]:
        brp.setBucketLength(bucketLength).setNumHashTables(numHashTables)
        model = brp.fit(df)
        score = evaluate_model(model, df)
        if score < best_score:
            best_score = score
            best_params = {"bucketLength": bucketLength, "numHashTables": numHashTables}
            best_model = model

# Print best parameters and their score
print(f"Best parameters: {best_params}")
print(f"Best score (avg Euclidean Distance): {best_score}")

# Stop Spark session
spark.stop()

