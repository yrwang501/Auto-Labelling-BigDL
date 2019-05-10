package com.intel.aimaster

import java.awt.image.DataBufferByte
import java.util.concurrent.atomic.AtomicInteger

import com.intel.analytics.bigdl.{DataSet, _}
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
import com.intel.analytics.bigdl.transform.vision.image.{FeatureTransformer, ImageFeature, ImageFrame}
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import javax.imageio.ImageIO
import org.apache.hadoop.hbase.client.Table
import org.apache.hadoop.hbase.util.Bytes
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import java.io.ByteArrayInputStream

import com.intel.analytics.bigdl.dataset.{DataSet, MiniBatch}
import org.apache.spark.rdd.RDD

import scala.util.Random

//import HBaseConnect.HBaseConnector._
import HBaseHelperAPI.HBaseHelperAPI._


object Train {
  LoggerFilter.redirectSparkInfoLogs()
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)
  Logger.getLogger("com.intel.analytics.bigdl.transform.vision.image").setLevel(Level.ERROR)
  val logger = Logger.getLogger(getClass)


  var coreSiteConf,hbaseSiteConf : String = _
  var hbaseconf : Broadcast[(String,String,String)] = _
  var table : Table = _
  var sc: SparkContext = _
  var trainRdd:RDD[ImageFeature] = _
  var valRdd:RDD[ImageFeature] = _
  var niter = 0
  var accScore = 0.0
  @volatile var isTraining = false
  @volatile var needsAbort = false

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
    val conf = Engine.createSparkConf().setAppName("Train ResNet on kfbio")
      .set("spark.rpc.message.maxSize", "200")
    sc = new SparkContext(conf)
    Engine.init
    println("===========Load module and connect to HBase table====================")

    //val table = connectToHBaseCached(args(0), args(1), args(3))
    coreSiteConf = readStrFromFile(coreSitePath)
    hbaseSiteConf = readStrFromFile(hbaseSitePath)
    hbaseconf = sc.broadcast((coreSiteConf, hbaseSiteConf, hbaseTableName))

    table = connectToHBaseCached(coreSiteConf, hbaseSiteConf, hbaseTableName)

    //make it a local val to be captured
    val _hbaseconf=hbaseconf
    //init hbase connections
    sc.parallelize(1 to engineNodeNumber(), engineNodeNumber()).map(idx => {
      val conf = _hbaseconf.value
      println(conf._1, conf._2, conf._3)
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

  def loadDataset(startRow:Int,stopRow:Int,batchSize:Int,validatePortition:Double) :Unit= {
    require(startRow<stopRow)
    // retrieve 3 columns from HBase (label, offset, image base64 string)
    val fetchBatchSize=30

    val _hbaseconf=hbaseconf
    val numSplits = divAndCeil(stopRow-startRow, fetchBatchSize)
    val rawDataset = sc.parallelize( 0 until numSplits, engineNodeNumber())
      .flatMap( arg => {
        val startIdx = startRow + arg * fetchBatchSize
        val endIdx = Math.min(startIdx+fetchBatchSize, stopRow)
        val strStart = Bytes.toBytes("%09d".format(startIdx))
        val strEnd = Bytes.toBytes("%09d".format(endIdx))


        val qualifiers = Array(col("data","data"),col("meta","label"),
          col("meta","train")
        )
        val conf = _hbaseconf.value
        val resultStringArray = scanGetFullDataRaw(connectToHBaseCached(conf._1, conf._2, conf._3),
          strStart, strEnd, endIdx - startIdx,  qualifiers)
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

    trainRdd = rawDataset.filter( data=> data._2!=0).map(tp=>tp._1).cache()
    //take 15% of the validation set.
    valRdd = rawDataset.filter(data=> data._2 == 0 ).map(tp=>tp._1).cache()

  }


  def doTrain(batchSize:Int,maxEpoch:Int,startRow:Int,stopRow:Int,
              modelSavingPath:String,checkpoint:Option[String]=None,stateSnapshot:Option[String]=None,modelSnapshot:Option[String]=None,
              classes:Int=2,depth:Int=50,validatePortition:Double=0.08,
              learningRate:Double=0.1,maxLr:Double=3.2,warmupEpoch:Int=1,
              weightDecay:Double=1e-4,momentum:Double=0.9,dampening:Double=0.0,
              nesterov:Boolean=true,optnet: Boolean = false, deltaHue:Double = 0.0, deltaContrast:Double = 1.0):Unit={////
    isTraining=true
    needsAbort=false
    val dataSetType = DatasetType.ImageNet

    val shortcut: ShortcutType = ShortcutType.B
    println(engineType())
    val model = if (modelSnapshot.isDefined) {
      Module.load[Float](modelSnapshot.get)
    } else {
      //        engineType() match {
      //         case MklBlas =>
      val curModel =
      ResNet(classNum = classes, T("shortcutType" -> shortcut, "depth" -> depth,
        "optnet" -> optnet, "dataSet" -> dataSetType))
      if (optnet) {
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
      /*case MklDnn =>
        nn.mkldnn.ResNet(param.batchSize / engineNodeNumber(), param.classes,
          T("depth" -> param.depth, "dataSet" -> ImageNet))*/
      //       }
    }
    val imageSize=224
    val trainDataSet = ImageNetDataSet2.trainD(ImageFrame.rdd(trainRdd), sc, imageSize, batchSize, deltaHue, deltaContrast)
    val validateSet = ImageNetDataSet2.valD(ImageFrame.rdd(valRdd), sc, imageSize, batchSize, validatePortition, deltaHue, deltaContrast)

    val optimMethod = if (stateSnapshot.isDefined) {
      val optim = OptimMethod.load[Float](stateSnapshot.get).asInstanceOf[SGD[Float]]
      val baseLr = learningRate
      val iterationsPerEpoch = math.ceil(1281167 / batchSize).toInt
      val warmUpIteration = iterationsPerEpoch * warmupEpoch
      val delta = (maxLr - baseLr) / warmUpIteration
      optim.learningRateSchedule = SGD.EpochDecayWithWarmUp(warmUpIteration, delta, imageNetDecay)
      optim
    } else {
      val baseLr = learningRate
      val iterationsPerEpoch = math.ceil(1281167 / batchSize).toInt
      val warmUpIteration = iterationsPerEpoch * warmupEpoch
      val delta = (maxLr - baseLr) / warmUpIteration

      logger.info(s"warmUpIteraion: $warmUpIteration, startLr: ${learningRate}, " +
        s"maxLr: $maxLr, " +
        s"delta: $delta, nesterov: ${nesterov}")
      new SGD[Float](learningRate = learningRate, learningRateDecay = 0.0,
        weightDecay = weightDecay, momentum = momentum, dampening = dampening,
        nesterov = nesterov,
        learningRateSchedule = SGD.EpochDecayWithWarmUp(warmUpIteration, delta, imageNetDecay))
    }

    val optimizer = Optimizer(
      model = model,
      dataset = trainDataSet,
      criterion = new CrossEntropyCriterion[Float]()
    )
    if (checkpoint.isDefined) {
      optimizer.setCheckpoint(checkpoint.get, Trigger.everyEpoch)
    }

    val logdir = "resnet-imagenet"
    val appName = s"${sc.applicationId}"
    val trainSummary = TrainSummary(logdir, appName)
    trainSummary.setSummaryTrigger("LearningRate", Trigger.severalIteration(1))
    trainSummary.setSummaryTrigger("Parameters", Trigger.severalIteration(10))
    val validationSummary = ValidationSummary(logdir, appName)

    var accSoreArray = Array[Double]()
    var accSoreHistory = Array[Double]()
    def avg(data: Seq[Double]) : Double={
      var sum=0.0
      data.foreach(v => sum += v)
      sum/data.length
    }

    def variance(data: Seq[Double]) : Double={
      var average=avg(data)
      var sum=0.0
      data.foreach(v => sum += (v-average)*(v-average))
      sum/data.length
    }

    def test(accSoreTemp: Double) : Double={
      //val accSoreTemp = state[Float]("score")
      var varianceArray=0.0
      if(accSoreArray.length<8){
        accSoreArray = accSoreArray:+accSoreTemp
        accScore = avg(accSoreArray)
        //varianceArray = variance(accSoreArray)
      }
      else
      {
        if(accSoreArray(0) < accSoreTemp){
          accSoreArray(0) = accSoreTemp
          accSoreArray = accSoreArray.sorted
        }
        accScore=avg(accSoreArray)
      }

      if(accSoreHistory.length<8){
        accSoreHistory = accSoreHistory:+accSoreTemp
        varianceArray = variance(accSoreHistory)
      }
      else
      {
        accSoreHistory(0)=accSoreHistory(1)
        accSoreHistory(1)=accSoreHistory(2)
        accSoreHistory(2)=accSoreHistory(3)
        accSoreHistory(3)=accSoreHistory(4)
        accSoreHistory(4)=accSoreHistory(5)
        accSoreHistory(5)=accSoreHistory(6)
        accSoreHistory(6)=accSoreHistory(7)
        accSoreHistory(7)=accSoreTemp
        varianceArray = variance(accSoreHistory)
      }
      println(s"variance: $varianceArray, accuracy: $accScore")
      varianceArray
    }

    val myTrigger = new Trigger {
      override def apply(state: utils.Table): Boolean = {
        niter = state[Int]("neval")
        val accSoreTemp = state[Float]("score")
        val varianceArray = if(niter % 5 ==0) {
          test(accSoreTemp)
        }
        else{
          1000.0
        }
        //accScore =
        //state[Int]("epoch") > maxEpoch || accScore > 0.85 || needsAbort
        state[Int]("epoch") > maxEpoch ||  needsAbort || (niter>15&&varianceArray<0.0001)

      }
    }
    val trainedModel = optimizer
      .setOptimMethod(optimMethod)
      .setValidation(Trigger.severalIteration(5),
        validateSet, Array(new Top1Accuracy[Float]))
      .setEndWhen(myTrigger)
      .optimize()

    //trainedModel.saveModule(modelSavingPath, overWrite = true)
    //trainedModel.save(param.modelSavingPath, true)
    isTraining=false

  }
  def main(args: Array[String]): Unit = {
    trainParser.parse(args, TrainParams()).foreach(param => {

      init(param.coreSitePath, param.hbaseSitePath, param.hbaseTableName)
      loadDataset(param.rowKeyStart.toInt,param.rowKeyEnd.toInt,param.batchSize,param.validatePortition)
      doTrain(param.batchSize,param.nepochs,param.rowKeyStart.toInt,param.rowKeyEnd.toInt,param.modelSavingPath,
        param.checkpoint,param.stateSnapshot,param.modelSnapshot,param.classes,param.depth,param.validatePortition,
        param.learningRate,param.maxLr,param.warmupEpoch,param.weightDecay,param.momentum,param.dampening,param.nesterov,param.optnet
      )
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
