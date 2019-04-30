$SPARK_HOME/bin/spark-submit \
--master "spark://ai-master-bigdl-0.sh.intel.com:7077" \
    --conf 'spark.driver.extraJavaOptions=-Dbigdl.engineType=mkldnn'\
    --conf 'spark.executor.extraJavaOptions=-Dbigdl.engineType=mkldnn' \
    --driver-memory 4g --class com.intel.aimaster.TrainServer \
	--total-executor-cores 72\
	--executor-cores 24\
        --executor-memory 40g\
   target/AI-Master-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
new-core-site.xml \
new-hbase-site.xml \
kfb_data \
1 \
2000 \
144 \
0.5 \
