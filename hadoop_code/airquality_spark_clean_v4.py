# -*- coding: utf-8 -*-
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
import sys

if len(sys.argv) != 3:
    print("Usage: spark-submit airquality_spark_clean_v3.py <INPUT_PATH> <OUTPUT_PATH>")
    sys.exit(1)

INPUT_PATH = sys.argv[1]
OUTPUT_PATH = sys.argv[2]

conf = SparkConf().setAppName("AirQuality Spark Cleaning v3")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

schema = StructType([
    StructField("region", StringType(), True),
    StructField("station_code", StringType(), True),
    StructField("station_name", StringType(), True),
    StructField("date_time", StringType(), True),
    StructField("SO2", DoubleType(), True),
    StructField("CO", DoubleType(), True),
    StructField("O3", DoubleType(), True),
    StructField("NO2", DoubleType(), True),
    StructField("PM10", DoubleType(), True),
    StructField("PM25", DoubleType(), True),
    StructField("address", StringType(), True),
])

def safe_float(v):
    if v is None:
        return None

    v = v.strip()

    if v == "" or v == "-1" or v == "-999":
        return None

    try:
        f = float(v)
        if f < 0:
            return None
        return f
    except:
        return None

def parse_line(line):
    parts = line.split(",")

    if len(parts) != 11:
        return None

    return (
        parts[0],
        parts[1],
        parts[2],
        parts[3],
        safe_float(parts[4]),
        safe_float(parts[5]),
        safe_float(parts[6]),
        safe_float(parts[7]),
        safe_float(parts[8]),
        safe_float(parts[9]),
        parts[10]
    )

print("Input path:", INPUT_PATH)
print("Output path:", OUTPUT_PATH)

raw = sc.textFile(INPUT_PATH)

header = raw.first()

data = raw.filter(lambda x: x != header and len(x.split(",")) == 11)

parsed = data.map(parse_line).filter(lambda x: x is not None)

df = sqlContext.createDataFrame(parsed, schema)

print("Schema:")
df.printSchema()

print("Sample data:")
df.show(5)

import subprocess

def hdfs_exists(path):
    return subprocess.call(["hdfs", "dfs", "-test", "-e", path]) == 0

if hdfs_exists(OUTPUT_PATH):
    print("Existing output detected, deleting:", OUTPUT_PATH)
    subprocess.call(["hdfs", "dfs", "-rm", "-r", OUTPUT_PATH])


print("Saving to Parquet...")
df.saveAsParquetFile(OUTPUT_PATH)

print("Completed clean parquet generation")

sc.stop()
