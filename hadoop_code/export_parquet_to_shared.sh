#!/bin/bash

###########################################
# Export normalized Parquet to VM Shared Folder
###########################################

HDFS_BASE="/user/airquality/zscore_parquet"
SHARED_BASE="/mnt/hgfs/csv/zscore_results"

echo "=================================="
echo " Parquet Export Pipeline Started"
echo "=================================="

mkdir -p ${SHARED_BASE}

# Get all year partitions
YEARS=$(hdfs dfs -ls ${HDFS_BASE} | grep "year=" | awk -F= '{print $2}')

for year in $YEARS
do
    echo "----------------------------------"
    echo "Exporting year: $year"
    echo "----------------------------------"

    HDFS_YEAR_PATH="${HDFS_BASE}/year=${year}"
    LOCAL_YEAR_PATH="${SHARED_BASE}/year_${year}"

    mkdir -p ${LOCAL_YEAR_PATH}

    # Directly copy parquet files
    hdfs dfs -get -f ${HDFS_YEAR_PATH}/*.parquet ${LOCAL_YEAR_PATH}/
    hdfs dfs -get -f ${HDFS_YEAR_PATH}/_SUCCESS ${LOCAL_YEAR_PATH}/ 2>/dev/null
    hdfs dfs -get -f ${HDFS_YEAR_PATH}/_metadata ${LOCAL_YEAR_PATH}/ 2>/dev/null
    hdfs dfs -get -f ${HDFS_YEAR_PATH}/_common_metadata ${LOCAL_YEAR_PATH}/ 2>/dev/null

    echo "Done: ${LOCAL_YEAR_PATH}"
done

echo "=================================="
echo " All Parquet Export Completed"
echo "=================================="
