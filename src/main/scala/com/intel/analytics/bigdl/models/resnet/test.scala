package com.intel.analytics.bigdl.models.resnet
import java.awt.image.{BufferedImage, DataBufferByte}

import org.apache.hadoop.hbase.util.Bytes
import HBaseHelperAPI.HBaseHelperAPI._
import com.intel.analytics.bigdl.nn.{Module, _}
import com.intel.analytics.bigdl.utils.{Engine, T}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image._
import javax.imageio.ImageIO
import org.apache.spark.SparkContext
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.transform.vision.image.augmentation.{Resize, _}
import java.io.{ByteArrayInputStream, File}

import org.apache.hadoop.hbase.client._

import scala.collection.JavaConversions._
import java.net._
import java.io._

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.optim.Predictor.predictSamples
import com.intel.analytics.bigdl.optim.{LocalPredictor, Predictor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.intermediate.IRGraph

import scala.reflect.ClassTag



object test {
  def modelProcessing[T](model: AbstractModule[Activity, Activity, T]): AbstractModule[Activity, Activity, T] = {
    val m = if (!model.isInstanceOf[Graph[T]]) model.toGraph() else model
    if (m.isInstanceOf[IRGraph[T]]) return m
    if (!m.isInstanceOf[StaticGraph[T]] || Engine.getEngineType() == "Mklblas") return model
    return m.asInstanceOf[StaticGraph[T]].toIRgraph().asInstanceOf[AbstractModule[Activity, Activity, T]]
  }




  def predictImage[T : ClassTag](model: Module[T],
                      imageFrame: ImageFrame,
                      broadcastModel: ModelBroadcast[T],
                   outputLayer: String = null,
                   shareBuffer: Boolean = false,
                   batchPerPartition: Int = 4,
                   predictKey: String = ImageFeature.predict,
                   featurePaddingParam: Option[PaddingParam[T]] = None) (implicit ev: TensorNumeric[T]): ImageFrame = {
    imageFrame match {
      case distributedImageFrame: DistributedImageFrame =>
        MyPredictor(model, featurePaddingParam, batchPerPartition)
          .predictImage(distributedImageFrame, broadcastModel, outputLayer, shareBuffer, predictKey)
      case localImageFrame: LocalImageFrame =>
        val predictor = LocalPredictor(model, featurePaddingParam, batchPerPartition)
        val imageFrame = predictor.predictImage(localImageFrame, outputLayer, shareBuffer,
          predictKey)
        predictor.shutdown()
        imageFrame
    }
  }

  def main(args: Array[String]): Unit = {
    if (args.length == 0) {
      println("arguments: core-site path, hbase-site path, model_path, table name")
    }else {
      for (arg <- args){
        println(arg)
      }
    }
    // create the socket server object
    val server = new ServerSocket(10001)
    val socketFinishLoading = server.accept()

    val conf = Engine.createSparkConf().setAppName("Test ResNet")
      .set("spark.akka.frameSize", 64.toString)
      .set("spark.task.maxFailures", "1")
    val sc = new SparkContext(conf)

    Engine.init
    println(Engine.getEngineType())

    println("===========Load module and connect to HBase table====================")


    //val table = connectToHBaseCached(args(0), args(1), args(3))
    val hbaseconf = sc.broadcast((args(0), args(1), args(3)))

    val loadAndBcastModel = () => {
      val model = test.modelProcessing(Module.loadModule(args(2)))
      //val model = Module.load(args(2))
      val bcastModel = MyPredictor.bcastModel(sc, model)

      sc.parallelize(1 to Engine.nodeNumber(), Engine.nodeNumber()).map( idx => {
        bcastModel.value()
        idx
      }).count()
      (model, bcastModel)
    }

    var (model, bcastModel) = loadAndBcastModel()

    //init hbase connections
    sc.parallelize(1 to Engine.nodeNumber(), Engine.nodeNumber()).map( idx => {
      val conf = hbaseconf.value
      connectToHBaseCached(conf._1, conf._2, conf._3)
      idx
    }).count()
    val table = connectToHBaseCached(args(0), args(1), args(3))

    println("============Finish loading===============")



    val outFinishLoading = new PrintWriter(socketFinishLoading.getOutputStream, true)
    outFinishLoading.println("Finish loading")
    socketFinishLoading.close()
    while(true) {
      println("Waiting for arguments: start row, stop row, number of rows")
      val socket = server.accept()
      // socket reader
      val in = new BufferedReader(new InputStreamReader(socket.getInputStream))
      // socket writer
      val out = new PrintWriter(socket.getOutputStream, true)

      val msgStr = in.readLine()

      msgStr match {
        case "stop" => // if the client sends "stop" message, then exits constant while loop and stop the spark context
          server.close()
          sc.stop()
          return
        case "reload" =>
          val (_model, _bcastModel) = loadAndBcastModel()
          model=_model
          bcastModel=_bcastModel
        case _ =>
          val msgArgs = msgStr.split(" ")
          for (arg <- msgArgs) {
            println(arg)
          }
          val toClient = "%d".format(msgArgs.length)
          //out.println(toClient)

          val startRow = msgArgs(0).toInt
          //val stopRow = Bytes.toBytes(msgArgs(1))
          val numOfImages = msgArgs(2).toInt
          val itmPerNode =  numOfImages/Engine.nodeNumber() + 1

//
          val pictureSplits = (0 until Engine.nodeNumber()).map(idx =>{
            val startIdx = startRow + idx * itmPerNode
            val endIdx = startRow + Math.min( (idx+1) * itmPerNode, numOfImages)
            (Bytes.toBytes("%09d".format(startIdx)),Bytes.toBytes("%09d".format(endIdx)), endIdx-startIdx)
          })
          val valRdd = sc.parallelize( pictureSplits , Engine.nodeNumber())
            .flatMap( arg => {
              val (startRow,stopRow,numOfImages) = arg
              val conf = hbaseconf.value
              val table = connectToHBaseCached(conf._1, conf._2, conf._3)

              val family = Bytes.toBytes("123_s20")
              val qualifiers = Array(Bytes.toBytes("data"))
              val ret = scanGetData(table, startRow, stopRow, numOfImages, family, qualifiers)
              //table.close()
              ret
          })

          //val numOfPredictions = retrievedStrings.length
          //println("predictions: " + numOfPredictions)

          // = sc.parallelize(retrievedStrings)

          val row, col = 4
          val eWidth, eHeight = 150
          val step = 120
          val numOfWindows = row * col

          val validateSet = valRdd.flatMap(rowString => {
            val rowKey: String = rowString.split(";")(0)

            val base64String: String = rowString.split(";")(1)
            val rawBytes: Array[Byte] = javax.xml.bind.DatatypeConverter
              .parseBase64Binary(base64String.map { case '-' => '+'; case '_' => '/'; case c => c })

            val bis = new ByteArrayInputStream(rawBytes)
            val image = ImageIO.read(bis)
            bis.close()

            val imf: Array[ImageFeature] = new Array[ImageFeature](numOfWindows)
            for (i <- 0 until numOfWindows) {
              imf(i) = new ImageFeature()
            }

            var y, x = 0
            for (i <- 0 until row) {
              x = 0
              for (j <- 0 until col) {
                val subImage = new BufferedImage(eWidth, eHeight, 5)
                val g = subImage.getGraphics
                g.drawImage(image.getSubimage(x, y, eWidth, eHeight), 0, 0, null)
                g.dispose()
                val bytes: Array[Byte] = subImage.getRaster.getDataBuffer.asInstanceOf[DataBufferByte].getData()
                val index = i * row + j

                //          val fileName = "/home/yilinma/Pictures/cropped_images/" + rowKey + "_" + index + ".jpg"
                //          ImageIO.write(subImage, "jpg", new File(fileName))


                // ImageFeature refers to the object "ImageFeature", which has some string fields
                // and then, call the imf(index)'s apply function
                imf(index)("bytes") = bytes
                imf(index)("originalSize") = (eWidth, eHeight, 3)
                imf(index)("rowKey") = rowKey
                imf(index)("index") = index
                imf(index)("x") = j * step
                imf(index)("y") = i * step
                x += step
              }
              y += step
            }
            imf
          })

          println("===========Print rdd====================")
          val repartitionedValidateSet = validateSet.coalesce(Engine.nodeNumber())
          println("Partitions:" + repartitionedValidateSet.partitions.size)
          // distributed image frame, which is going to be sent to the model
          val distributedImageFrame = ImageFrame.rdd(repartitionedValidateSet) ->
            PixelBytesToMat() ->
            Resize(256, 256) ->
            CenterCrop(224, 224) ->
            ChannelScaledNormalizer(104, 117, 123, 0.0078125) ->
            MatToTensor() -> ImageFrameToSample()

          println("===========transform over====================")
          println("===========Predice Image====================")
          val result = predictImage(model, distributedImageFrame, bcastModel).toDistributed()

          println("====================select keys================")
          val keys = result.rdd.map(r => {
            val rowKey = r("rowKey").asInstanceOf[String]
            val x = r("x").asInstanceOf[Int]
            val y = r("y").asInstanceOf[Int]
            val pred = r("predict").asInstanceOf[Tensor[Float]]
            val index = r("index").asInstanceOf[Int]
            Map("rowKey" -> rowKey, "index" -> index, "x" -> x, "y" -> y, "predict" -> pred)
          })
          println("=================Print Result=======================")
          val features = keys.collect()


          for (f <- features) {
            //println(f("rowKey"))
            //println(f("index"))
            //println(f("x").toString + "," + f("y").toString)
            //println(f("predict"))
            val key: String = f("rowKey").asInstanceOf[String]
            val ind: String = f("index").toString

            val resultTensor = f("predict").asInstanceOf[Tensor[Float]]
            val resultArray = resultTensor.toArray()

            if (resultArray(0) < resultArray(1)) {
              //println("positive")
            } else {
              //println("negative")
            }
            // positive: booleanLabel == True, negative: booleanLabel == False
            //      val booleanLabel = resultTensor.valueAt(1) > resultTensor.valueAt(0)
            //      val intLabel = if (booleanLabel) 1 else 0
            //      println("Label: ", intLabel)
            //      println("tensor: " + resultTensor)
            //      val softMaxLayer = SoftMax()
            //      val output = softMaxLayer.forward(resultTensor)
            //      println("softmax: " + output)

          }

          val numOfPredictions = features.size
          println("==================Start Putting===================")
          var putList = new java.util.ArrayList[Put]()
          for (i <- 0 until numOfPredictions) {
            var label = 0
            var logitMax = 0.0
            var xCoordinate = 0
            var yCoordinate = 0
            val curRowKey: String = features(i * numOfWindows)("rowKey").asInstanceOf[String]
            for (j <- 0 until numOfWindows) {
              val feature = features(i * numOfWindows + j)
              val resultArray = feature("predict").asInstanceOf[Tensor[Float]].toArray()
              if (resultArray(0) < resultArray(1)) {
                // positive
                label += 1
                if (resultArray(1) > logitMax) {
                  logitMax = resultArray(1)
                  xCoordinate = feature("x").asInstanceOf[Int]
                  yCoordinate = feature("y").asInstanceOf[Int]
                }

              }
            }
            val putObj = new Put(Bytes.toBytes(curRowKey))
            if (label > 0) {
              // if label > 0, put (xCoordinate, yCoordinate), pos 1
              putObj.addColumn(Bytes.toBytes("123_s20"), Bytes.toBytes("pos"), Bytes.toBytes("1"))
              putObj.addColumn(Bytes.toBytes("123_s20"), Bytes.toBytes("offset"), Bytes.toBytes(xCoordinate.toString + "," + yCoordinate))
            }
            else {
              // else put # pos 0
              putObj.addColumn(Bytes.toBytes("123_s20"), Bytes.toBytes("pos"), Bytes.toBytes("0"))
              putObj.addColumn(Bytes.toBytes("123_s20"), Bytes.toBytes("offset"), Bytes.toBytes("#"))
            }

            putList += putObj
          }
          table.put(putList)
          table.close()
          println("==================Stop putting===================")
          // go back to Django process
          out.println(toClient)
          socket.close()
      }

    } //end match
  }//end while

}
