$SPARK_HOME/bin/spark-submit \
--master "spark://ai-master-bigdl-0.sh.intel.com:7077" \
    --driver-memory 4g --class com.intel.aimaster.Train \
	--total-executor-cores 48\
	--executor-cores 24\
        --executor-memory 40g\
   target/AI-Master-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
--batchSize 96 --nEpochs 3 --learningRate 0.1 --warmupEpoch 1 \
--maxLr 3.2 --depth 50 --classes 2 \
--coreSitePath new-core-site.xml \
--hbaseSitePath new-hbase-site.xml \
--hbaseTableName kfb_data --rowKeyStart 000000001 --rowKeyEnd 000040786 \
--modelSavingPath MKL_DNN.obj
