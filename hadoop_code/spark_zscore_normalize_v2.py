# -*- coding: utf-8 -*-
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import sys

if len(sys.argv) != 3:
    print("Usage: spark-submit spark_zscore_normalize.py <INPUT_PARQUET> <OUTPUT_PARQUET>")
    sys.exit(1)

INPUT_PATH = sys.argv[1]
OUTPUT_PATH = sys.argv[2]

conf = SparkConf().setAppName("AirQuality Z-score Normalization")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

print("Input path:", INPUT_PATH)
print("Output path:", OUTPUT_PATH)

df = sqlContext.parquetFile(INPUT_PATH)

value_cols = ["SO2", "CO", "O3", "NO2", "PM10", "PM25"]

stats = {}

for col in value_cols:
    rdd = df.select(col).dropna().rdd.map(lambda r: r[0])
    count = rdd.count()

    if count == 0:
        print(col, "has no valid values")
        continue

    mean = rdd.mean()
    std = rdd.stdev()

    stats[col.lower() + "_mean"] = mean
    stats[col.lower() + "_std"] = std

print("Statistics:", stats)

def z_expr(col):
    key_mean = col.lower() + "_mean"
    key_std = col.lower() + "_std"

    if key_mean not in stats or key_std not in stats:
        return col

    mean = stats[key_mean]
    std = stats[key_std]

    if std == 0 or std is None:
        return col

    return "(%s - %f) / %f AS %s_z" % (col, mean, std, col.lower())

df.registerTempTable("cleaned")

query = """
SELECT
    region,
    station_code,
    station_name,
    date_time,
    {so2},
    {co},
    {o3},
    {no2},
    {pm10},
    {pm25},
    address,
    substring(date_time, 0, 4) AS year
FROM cleaned
""".format(
    so2=z_expr("SO2"),
    co=z_expr("CO"),
    o3=z_expr("O3"),
    no2=z_expr("NO2"),
    pm10=z_expr("PM10"),
    pm25=z_expr("PM25")
)

df_z = sqlContext.sql(query)

print("Schema after z-score:")
df_z.printSchema()

df_z.show(5)

print("Saving z-score parquet...")
df_z.saveAsParquetFile(OUTPUT_PATH)

sc.stop()
