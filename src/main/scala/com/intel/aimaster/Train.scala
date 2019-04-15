package com.intel.aimaster

import java.awt.image.DataBufferByte
import java.util.concurrent.atomic.AtomicInteger

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.models.inception.{ImageNet2012, ImageNet2012Val}
import com.intel.analytics.bigdl.models.resnet.{ImageNetDataSet2, ResNet, Utils2}
import com.intel.analytics.bigdl.models.resnet.ResNet.{DatasetType, ShortcutType}
import com.intel.analytics.bigdl.models.resnet.TrainKfbio.getClass
import com.intel.analytics.bigdl.models.resnet.Utils2.{TrainParams, trainParser}
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.mkldnn.ResNet.DatasetType.ImageNet
import com.intel.analytics.bigdl.nn.{BatchNormalization, Container, CrossEntropyCriterion, Module}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric._
import com.intel.analytics.bigdl.transform.vision.image.{ImageFeature, ImageFrame}
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import javax.imageio.ImageIO
import org.apache.hadoop.hbase.client.Table
import org.apache.hadoop.hbase.util.Bytes
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import java.io.ByteArrayInputStream

import scala.util.Random

//import HBaseConnect.HBaseConnector._
import HBaseHelperAPI.HBaseHelperAPI._


object Train {
  LoggerFilter.redirectSparkInfoLogs()
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)
  val logger = Logger.getLogger(getClass)


  var coreSiteConf,hbaseSiteConf : String = _
  var hbaseconf : Broadcast[(String,String,String)] = _
  var table : Table = _
  var sc: SparkContext = _

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


  def readStrFromFile(path: String): String = {
    val source = scala.io.Source.fromFile(path)
    val lines = try source.mkString finally source.close()
    lines
  }

  def init(coreSitePath: String, hbaseSitePath: String,  hbaseTableName: String): Unit = {
    println("===========Load module and connect to HBase table====================")

    //val table = connectToHBaseCached(args(0), args(1), args(3))
    coreSiteConf = readStrFromFile(coreSitePath)
    hbaseSiteConf = readStrFromFile(hbaseSitePath)
    hbaseconf = sc.broadcast((coreSiteConf, hbaseSiteConf, hbaseTableName))

    table = connectToHBaseCached(coreSiteConf, hbaseSiteConf, hbaseTableName)

    //make it a local val to be captured
    val _hbaseconf=hbaseconf
    //init hbase connections
    sc.parallelize(1 to engineCoreNumber(), engineCoreNumber()).map(idx => {
      val conf = _hbaseconf.value
      table = connectToHBaseCached(conf._1, conf._2, conf._3)
      idx
    }).count()


    println("============Finish loading===============")

  }

  def divAndCeil(num: Int, divisor: Int): Int = (num + divisor - 1) / divisor

  def col(family: String, colname: String) : (Array[Byte],Array[Byte])={
    (Bytes.toBytes(family), Bytes.toBytes(colname))
  }

  import java.awt.Graphics2D
  import java.awt.Image
  import java.awt.image.BufferedImage

  def resizeImage(img: BufferedImage, newW: Int, newH: Int): BufferedImage = {
    val tmp = img.getScaledInstance(newW, newH, Image.SCALE_SMOOTH)
    val dimg = new BufferedImage(newW, newH, BufferedImage.TYPE_3BYTE_BGR)
    val g2d = dimg.createGraphics
    g2d.drawImage(tmp, 0, 0, null)
    g2d.dispose()
    dimg
  }

  def main(args: Array[String]): Unit = {
    trainParser.parse(args, TrainParams()).map(param => {
      val conf = Engine.createSparkConf().setAppName("Train ResNet on kfbio")
        .set("spark.rpc.message.maxSize", "200")
      sc = new SparkContext(conf)
      Engine.init
      init(param.coreSitePath, param.hbaseSitePath, param.hbaseTableName)

      val batchSize = param.batchSize
      val (imageSize, dataSetType, maxEpoch, dataSet) =
        (224, DatasetType.ImageNet, param.nepochs, ImageNetDataSet2)

      //val trainDataSet = dataSet.trainDataSet(param.folder + "/train", sc, imageSize, batchSize)
      //val validateSet = dataSet.valDataSet(param.folder + "/val", sc, imageSize, batchSize)

      val startRow = param.rowKeyStart.toInt
      val stopRow = param.rowKeyEnd.toInt
      require(startRow<stopRow)
      // retrieve 3 columns from HBase (label, offset, image base64 string)
      val fetchBatchSize=30

      val _hbaseconf=hbaseconf
      val numSplits = divAndCeil(stopRow-startRow, fetchBatchSize)
      val rawDataset = sc.parallelize( 0 until numSplits)
        .flatMap( arg => {
          val startIdx = startRow + arg * fetchBatchSize
          val endIdx = Math.min(startIdx+fetchBatchSize, stopRow)
          val strStart = Bytes.toBytes("%09d".format(startIdx))
          val strEnd = Bytes.toBytes("%09d".format(endIdx))


          val qualifiers = Array(col("data","data"),col("meta","label"),
            col("meta","train")
          )

          val resultStringArray = scanGetFullDataRaw(table, strStart, strEnd, endIdx - startIdx,  qualifiers)
          //table.close()
          resultStringArray.map(result => {
            val data = result.getValue(qualifiers(0)._1, qualifiers(0)._2)
            val label = Bytes.toString(result.getValue(qualifiers(1)._1, qualifiers(1)._2))
            val isTrain = Bytes.toString(result.getValue(qualifiers(2)._1, qualifiers(2)._2))
            (data, label.toFloat, isTrain.toInt)
          })
        }).map(tp=>{

        val pixelArr = new ByteArrayInputStream(tp._1)
        val image = ImageIO.read(pixelArr)

        val label = Tensor[Float](T(tp._2 + 1))
        val imf = new ImageFeature()
        val rawdata = image.getRaster.getDataBuffer.asInstanceOf[DataBufferByte].getData()

        imf(ImageFeature.bytes) = rawdata
        imf(ImageFeature.label) = label
        imf(ImageFeature.originalSize) = (image.getWidth, image.getHeight, 3)
        (imf, tp._3)
      })

      val trainRdd = rawDataset.filter( data=> data._2!=0).map(tp=>tp._1).cache()
      val valRdd = rawDataset.filter(data=> data._2==0 ).map(tp=>tp._1).cache()
      val trainDataSet = dataSet.trainD(ImageFrame.rdd(trainRdd), sc, imageSize, batchSize)
      val validateSet = dataSet.valD(ImageFrame.rdd(valRdd), sc, imageSize, batchSize)
      //println(s"size $trSz - $vaSz")
      val shortcut: ShortcutType = ShortcutType.B
      println(engineType())
      val model = if (param.modelSnapshot.isDefined) {
        Module.load[Float](param.modelSnapshot.get)
      } else {
        engineType() match {
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
            val parallisim = engineCoreNumber()
            setParallism(curModel, parallisim)

            curModel
          case MklDnn =>
            nn.mkldnn.ResNet(param.batchSize / engineNodeNumber(), param.classes,
              T("depth" -> param.depth, "dataSet" -> ImageNet))
        }
      }


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
    if (model.isInstanceOf[BatchNormalization[Float]]) {
      model.asInstanceOf[BatchNormalization[Float]].setParallism(parallism)
    }
    if(model.isInstanceOf[Container[_, _, Float]]) {
      model.asInstanceOf[Container[_, _, Float]].
        modules.foreach(sub => setParallism(sub, parallism))
    }
  }
}
