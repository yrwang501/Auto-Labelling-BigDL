/usr/local/spark/spark-2.1.0-bin-hadoop2.6/bin/spark-submit --master "spark://wyr-HP-EliteDesk-800-G1-TWR.sh.intel.com:7077" \
    --conf 'spark.driver.extraJavaOptions=-Dbigdl.engineType=mkldnn'\
    --conf 'spark.executor.extraJavaOptions=-Dbigdl.engineType=mkldnn' \
    --driver-memory 2g --class com.intel.aimaster.PredictorServer \
	--total-executor-cores 4\
	--executor-cores 4\
    target/AI-Master-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
	core-site.xml \
	hbase-site.xml \
	hdfs://ai-master-bigdl-0.sh.intel.com:8020/model_new_helper_API_10.obj \
	kfb_512_100_test \
	http://localhost:13345/updateStatus

