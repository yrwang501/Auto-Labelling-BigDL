$SPARK_HOME/bin/spark-submit --master "spark://localhost:7077" \
    --conf 'spark.driver.extraJavaOptions=-Dbigdl.engineType=mkldnn'\
    --conf 'spark.executor.extraJavaOptions=-Dbigdl.engineType=mkldnn' \
    --driver-memory 8g --class com.intel.analytics.bigdl.models.resnet.test \
	--total-executor-cores 4\
	--executor-cores 4\
    target/AI-Master-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
	/home/menooker/AI-Master/configs_core_and_model/core-site.xml \
	/home/menooker/AI-Master/configs_core_and_model/hbase-site.xml \
	/home/menooker/AI-Master/configs_core_and_model/model_new_helper_API_10.obj \
	kfb_512_100_test
