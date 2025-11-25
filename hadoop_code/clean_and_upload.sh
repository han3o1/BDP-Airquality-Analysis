#!/bin/bash

BASE_INPUT="/mnt/hgfs/csv"
HDFS_BASE="/user/airquality/data"
SCRIPT="/data/clean_only.py"

for YEAR_DIR in ${BASE_INPUT}/20*
do
    YEAR=$(basename $YEAR_DIR)

    echo "=========================="
    echo "Year: $YEAR"
    echo "=========================="

    INPUT_DIR="${BASE_INPUT}/${YEAR}"
    TEMP_DIR="${BASE_INPUT}/${YEAR}_clean"
    HDFS_YEAR="${HDFS_BASE}/year=${YEAR}"

    mkdir -p $TEMP_DIR

    for file in ${INPUT_DIR}/${YEAR}-*.csv
    do
        base=$(basename $file .csv)
        month=${base:5:2}

        echo "  Processing $file (month=$month)"

        CLEAN_FILE="${TEMP_DIR}/${base}.clean.csv"
        HDFS_DIR="${HDFS_YEAR}/month=${month}"

        python $SCRIPT $file $CLEAN_FILE

        hdfs dfs -mkdir -p $HDFS_DIR
        hdfs dfs -put -f $CLEAN_FILE ${HDFS_DIR}/
    done
done

echo "All years upload completed."
