package com.intel.analytics.bigdl.models.resnet

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.models.inception.{ImageNet2012, ImageNet2012Val}
import com.intel.analytics.bigdl.models.resnet.ResNet.{DatasetType, ShortcutType}
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.nn.mkldnn.ResNet.DatasetType.ImageNet
import com.intel.analytics.bigdl.nn.{BatchNormalization, Container, CrossEntropyCriterion, Module}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric._
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import org.apache.hadoop.hbase.util.Bytes
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

//import HBaseConnect.HBaseConnector._
import HBaseHelperAPI.HBaseHelperAPI._


object TrainKfbio {
  LoggerFilter.redirectSparkInfoLogs()
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)
  val logger = Logger.getLogger(getClass)

  import Utils2._

  def imageNetDecay(epoch: Int): Double = {
    if (epoch >= 80) {
      3
    } else if (epoch >= 60) {
      2
    } else if (epoch >= 30) {
      1
    } else {
      0.0
    }
  }

  def main(args: Array[String]): Unit = {
    trainParser.parse(args, TrainParams()).map(param => {
      val conf = Engine.createSparkConf().setAppName("Train ResNet on kfbio")
        .set("spark.rpc.message.maxSize", "200")
      val sc = new SparkContext(conf)
      Engine.init

      val batchSize = param.batchSize
      val (imageSize, dataSetType, maxEpoch, dataSet) =
        (224, DatasetType.ImageNet, param.nepochs, ImageNetDataSet2)

      //val trainDataSet = dataSet.trainDataSet(param.folder + "/train", sc, imageSize, batchSize)
      //val validateSet = dataSet.valDataSet(param.folder + "/val", sc, imageSize, batchSize)


      val table = connectToHBase(param.coreSitePath, param.hbaseSitePath, param.hbaseTableName)
      val startRow = Bytes.toBytes(param.rowKeyStart)
      val stopRow = Bytes.toBytes(param.rowKeyEnd)
      val family = Bytes.toBytes("123_s20")
      // retrieve 3 columns from HBase (label, offset, image base64 string)
      val qualifiers = Array(Bytes.toBytes("pos"), Bytes.toBytes("offset"), Bytes.toBytes("data"))
      val resultStringArray = scanGetFullData(table, startRow, stopRow, 30, family, qualifiers)
      val trainRdd = sc.parallelize(resultStringArray)
      val valRdd = sc.parallelize(resultStringArray)


      val trainDataSet = dataSet.trainD(trainRdd, sc, imageSize, batchSize)
      val validateSet = dataSet.valD(valRdd, sc, imageSize, batchSize)

      val shortcut: ShortcutType = ShortcutType.B
      println(Engine.getEngineType())
      val model = if (param.modelSnapshot.isDefined) {
        Module.load[Float](param.modelSnapshot.get)
      } else {
        Engine.getEngineType() match {
          case MklBlas =>
            val curModel =
              ResNet(classNum = param.classes, T("shortcutType" -> shortcut, "depth" -> param.depth,
                "optnet" -> param.optnet, "dataSet" -> dataSetType))
            if (param.optnet) {
              ResNet.shareGradInput(curModel)
            }
            ResNet.modelInit(curModel)

            /* Here we set parallism specificall for BatchNormalization and its Sub Layers, this is
            very useful especially when you want to leverage more computing resources like you want
            to use as many cores as possible but you cannot set batch size too big for each core due
            to the memory limitation, so you can set batch size per core smaller, but the smaller
            batch size will increase the instability of convergence, the synchronization among BN
            layers basically do the parameters synchronization among cores and thus will avoid the
            instability while improves the performance a lot. */
            val parallisim = Engine.coreNumber
            setParallism(curModel, parallisim)

            curModel
          case MklDnn =>
            nn.mkldnn.ResNet(param.batchSize / Engine.nodeNumber(), param.classes,
              T("depth" -> 50, "dataSet" -> ImageNet))
        }
      }

      println(model)

      val optimMethod = if (param.stateSnapshot.isDefined) {
        val optim = OptimMethod.load[Float](param.stateSnapshot.get).asInstanceOf[SGD[Float]]
        val baseLr = param.learningRate
        val iterationsPerEpoch = math.ceil(1281167 / param.batchSize).toInt
        val warmUpIteration = iterationsPerEpoch * param.warmupEpoch
        val maxLr = param.maxLr
        val delta = (maxLr - baseLr) / warmUpIteration
        optim.learningRateSchedule = SGD.EpochDecayWithWarmUp(warmUpIteration, delta, imageNetDecay)
        optim
      } else {
        val baseLr = param.learningRate
        val iterationsPerEpoch = math.ceil(1281167 / param.batchSize).toInt
        val warmUpIteration = iterationsPerEpoch * param.warmupEpoch
        val maxLr = param.maxLr
        val delta = (maxLr - baseLr) / warmUpIteration

        logger.info(s"warmUpIteraion: $warmUpIteration, startLr: ${param.learningRate}, " +
          s"maxLr: $maxLr, " +
          s"delta: $delta, nesterov: ${param.nesterov}")
        new SGD[Float](learningRate = param.learningRate, learningRateDecay = 0.0,
          weightDecay = param.weightDecay, momentum = param.momentum, dampening = param.dampening,
          nesterov = param.nesterov,
          learningRateSchedule = SGD.EpochDecayWithWarmUp(warmUpIteration, delta, imageNetDecay))
      }

      val optimizer = Optimizer(
        model = model,
        dataset = trainDataSet,
        criterion = new CrossEntropyCriterion[Float]()
      )
      if (param.checkpoint.isDefined) {
        optimizer.setCheckpoint(param.checkpoint.get, Trigger.everyEpoch)
      }

      val logdir = "resnet-imagenet"
      val appName = s"${sc.applicationId}"
      val trainSummary = TrainSummary(logdir, appName)
      trainSummary.setSummaryTrigger("LearningRate", Trigger.severalIteration(1))
      trainSummary.setSummaryTrigger("Parameters", Trigger.severalIteration(10))
      val validationSummary = ValidationSummary(logdir, appName)

      val trainedModel = optimizer
        .setOptimMethod(optimMethod)
        .setValidation(Trigger.everyEpoch,
          validateSet, Array(new Top1Accuracy[Float]))
        .setEndWhen(Trigger.maxEpoch(maxEpoch))
        .optimize()

      trainedModel.saveModule(param.modelSavingPath, overWrite = true)
      //trainedModel.save(param.modelSavingPath, true)

      sc.stop()
    })
  }

  private def setParallism(model: AbstractModule[_, _, Float], parallism: Int): Unit = {
    model match {
      case value: BatchNormalization[Float] =>
        value.setParallism(parallism)
      case value: Container[_, _, Float] =>
        value.modules.foreach(sub => setParallism(sub, parallism))
      case _ =>
    }
  }
}
