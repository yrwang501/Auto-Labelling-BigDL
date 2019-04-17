$SPARK_HOME/bin/spark-submit --master "spark://ai-master-bigdl-0.sh.intel.com:7077" \
    --conf 'spark.driver.extraJavaOptions=-Dbigdl.engineType=mkldnn'\
    --conf 'spark.executor.extraJavaOptions=-Dbigdl.engineType=mkldnn' \
        --executor-memory 40g\
         --driver-memory 4g --class com.intel.aimaster.TrainServer \
        --total-executor-cores 48\
        --executor-cores 24\
    target/AI-Master-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
        new-core-site.xml \
        new-hbase-site.xml \
        kfb_data \
        1 40000 96 0.05

