#!/bin/bash

###########################################
# Full Auto AirQuality Hadoop Spark Pipeline
# - Auto detect years from HDFS
# - Run cleaning + z-score normalization
###########################################

CLEAN_SCRIPT="/data/airquality_spark_clean_v4.py"
ZSCORE_SCRIPT="/data/spark_zscore_normalize_v3.py"

BASE_INPUT="/user/airquality/data"
BASE_CLEAN="/user/airquality/clean_parquet"
BASE_ZSCORE="/user/airquality/zscore_parquet"

PYTHON_EXEC="python2.6"

echo "===================================="
echo " Starting AirQuality Full Pipeline "
echo "===================================="

# 1. detect available years automatically
YEARS=$(hdfs dfs -ls ${BASE_INPUT} \
    | awk '{print $8}' \
    | grep "year=" \
    | awk -F= '{print $2}')

echo "Detected years:"
echo "$YEARS"
echo ""

# 2. loop for each year
for year in $YEARS
do
    echo "------------------------------------"
    echo "Processing year: $year"
    echo "------------------------------------"

    INPUT_PATH="hdfs:///user/airquality/data/year=${year}/*/*.csv"
    CLEAN_PATH="hdfs:///user/airquality/clean_parquet/year=${year}"
    ZSCORE_PATH="hdfs:///user/airquality/zscore_parquet/year=${year}"

    echo "[1] Cleaning Raw Data -> Parquet"
    spark-submit \
        --master yarn \
        --deploy-mode client \
        --conf spark.pyspark.python=${PYTHON_EXEC} \
        ${CLEAN_SCRIPT} \
        ${INPUT_PATH} \
        ${CLEAN_PATH}

    echo "[2] Z-score Normalization"
    spark-submit \
        --master yarn \
        --deploy-mode client \
        --conf spark.pyspark.python=${PYTHON_EXEC} \
        ${ZSCORE_SCRIPT} \
        ${CLEAN_PATH} \
        ${ZSCORE_PATH}

    echo "Year $year completed."
done

echo "===================================="
echo " Pipeline Completed Successfully "
echo "===================================="

/data/export_parquet_to_shared.sh
