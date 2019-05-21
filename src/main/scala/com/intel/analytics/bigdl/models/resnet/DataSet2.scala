
package com.intel.analytics.bigdl.models.resnet

import com.intel.analytics.bigdl.{DataSet, dataset}
import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.dataset.image._
import com.intel.analytics.bigdl.dataset.image.{HFlip => JHFlip}
import com.intel.analytics.bigdl.dataset.DataSet2.SeqFileFolder2
import com.intel.analytics.bigdl.transform.vision
import com.intel.analytics.bigdl.transform.vision.image._
import com.intel.analytics.bigdl.transform.vision.image.augmentation.{Contrast, _}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.bigdl.transform.vision.image.{FeatureTransformer, ImageFeature}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import java.util
import org.opencv.core.{Core, Mat}
import scala.util.Random

/**
  * define some resnet datasets: trainDataSet and valDataSet.
  */
trait ResNetDataSet {
  def trainDataSet(path: String, batchSize: Int, size: Int)
  : DataSet[MiniBatch[Float]]
  def valDataSet(path: String, batchSize: Int, size: Int)
  : DataSet[MiniBatch[Float]]
  def valDataSet(path: String, sc: SparkContext, imageSize: Int, batchSize: Int)
  : DataSet[MiniBatch[Float]]
  def trainDataSet(path: String, sc: SparkContext, imageSize: Int, batchSize: Int)
  : DataSet[MiniBatch[Float]]
}

object Cifar10DataSet extends ResNetDataSet {

  val trainMean = (0.4913996898739353, 0.4821584196221302, 0.44653092422369434)
  val trainStd = (0.24703223517429462, 0.2434851308749409, 0.26158784442034005)
  val testMean = (0.4942142913295297, 0.4851314002725445, 0.45040910258647154)
  val testStd = (0.2466525177466614, 0.2428922662655766, 0.26159238066790275)

  override def trainDataSet(path: String, batchSize: Int, size: Int)
  : DataSet[MiniBatch[Float]] = {

   DataSet.array(Utils2.loadTrain(path))
      .transform(BytesToBGRImg())
      .transform(BGRImgNormalizer(trainMean, trainStd))
      .transform(JHFlip(0.5))
      .transform(BGRImgRdmCropper(cropWidth = 32, cropHeight = 32, padding = 4))
      .transform(BGRImgToBatch(batchSize))
  }

  override def valDataSet(path: String, batchSize: Int, size: Int)
  : DataSet[MiniBatch[Float]] = {

   DataSet.array(Utils2.loadTest(path))
      .transform(BytesToBGRImg())
      .transform(BGRImgNormalizer(testMean, testStd))
      .transform(BGRImgToBatch(batchSize))
  }

  override def valDataSet(path: String, sc: SparkContext, imageSize: Int, batchSize: Int)
  : DataSet[MiniBatch[Float]] = {

   DataSet.array(Utils2.loadTest(path), sc)
      .transform(BytesToBGRImg())
      .transform(BGRImgNormalizer(trainMean, trainStd))
      .transform(BGRImgToBatch(batchSize))
  }

  override def trainDataSet(path: String, sc: SparkContext, imageSize: Int, batchSize: Int)
  : DataSet[MiniBatch[Float]] = {

   DataSet.array(Utils2.loadTrain(path), sc)
      .transform(BytesToBGRImg())
      .transform(BGRImgNormalizer(testMean, testStd))
      .transform(JHFlip(0.5))
      .transform(BGRImgRdmCropper(cropWidth = 32, cropHeight = 32, padding = 4))
      .transform(BGRImgToBatch(batchSize))
  }
}

object ImageNetDataSet2 extends ResNetDataSet {

  val trainMean = (0.485, 0.456, 0.406)
  val trainStd = (0.229, 0.224, 0.225)
  val testMean = trainMean
  val testStd = trainStd

  override def trainDataSet(path: String, batchSize: Int, size: Int)
  : DataSet[MiniBatch[Float]] = {

   DataSet.array(Utils2.loadTrain(path))
      .transform(BytesToBGRImg())
      .transform(BGRImgNormalizer(trainMean, trainStd))
      .transform(JHFlip(0.5))
      .transform(BGRImgRdmCropper(cropWidth = 32, cropHeight = 32, padding = 4))
      .transform(BGRImgToBatch(batchSize))
  }

  override def valDataSet(path: String, batchSize: Int, size: Int)
  : DataSet[MiniBatch[Float]] = {

   DataSet.array(Utils2.loadTest(path))
      .transform(BytesToBGRImg())
      .transform(BGRImgNormalizer(testMean, testStd))
      .transform(BGRImgToBatch(batchSize))
  }

  override def valDataSet(path: String, sc: SparkContext, imageSize: Int, batchSize: Int)
  : DataSet[MiniBatch[Float]] = {
    SeqFileFolder2.filesToImageFeatureDataset(path, sc, 2).transform(
      MTImageFeatureToBatch(
        width = imageSize,
        height = imageSize,
        batchSize = batchSize,
        transformer = PixelBytesToMat() ->
          RandomResize(256, 256) ->
          RandomCropper(224, 224, false, CropCenter) ->
          ChannelScaledNormalizer(104, 117, 123, 0.0078125) ->
          MatToTensor[Float](), toRGB = false
      )
    )
  }

  override def trainDataSet(path: String, sc: SparkContext, imageSize: Int, batchSize: Int)
  : DataSet[MiniBatch[Float]] = {
    SeqFileFolder2.filesToImageFeatureDataset(path, sc, 2).transform(
      MTImageFeatureToBatch(
        width = imageSize,
        height = imageSize,
        batchSize = batchSize,
        transformer = PixelBytesToMat() ->
          RandomAlterAspect() ->
          RandomCropper(224, 224, true, CropRandom) ->
          ChannelScaledNormalizer(104, 117, 123, 0.0078125) ->
          MatToTensor[Float](), toRGB = false
      )
    )
  }

  def valD(rdd: RDD[String], sc: SparkContext, imageSize: Int, batchSize: Int)
  : DataSet[MiniBatch[Float]] = {
    SeqFileFolder2.rddToImageFeatureDataset(rdd, sc, 2).transform(
      MTImageFeatureToBatch(
        width = imageSize,
        height = imageSize,
        batchSize = batchSize,
        transformer = PixelBytesToMat() ->
          RandomResize(256, 256) ->
          RandomCropper(224, 224, true, CropRandom) ->
          ChannelScaledNormalizer(104, 117, 123, 0.0078125) ->
          MatToTensor[Float](), toRGB = false
      )
    )
  }

  def trainD(rdd: RDD[String], sc: SparkContext, imageSize: Int, batchSize: Int)
  : DataSet[MiniBatch[Float]] = {
    SeqFileFolder2.rddToImageFeatureDataset(rdd, sc, 2).transform(
      MTImageFeatureToBatch(
        width = imageSize,
        height = imageSize,
        batchSize = batchSize,
        transformer = PixelBytesToMat() ->
          RandomAlterAspect() ->
          RandomCropper(224, 224, true, CropRandom) ->
          ChannelScaledNormalizer(104, 117, 123, 0.0078125) ->
          MatToTensor[Float](), toRGB = false
      )
    )
  }

  /**
    * Adjust the image contrast
    * @param deltaLow RandRotation parameter low bound
    * @param deltaHigh RandRotation parameter high bound
    */
  class RandRotation(deltaLow: Double, deltaHigh: Double)//0-1 1-2 2-3 3-4
    extends FeatureTransformer {

    require(deltaHigh <= 4&& deltaLow>=0, "RandRotation lower and upper must be in limit.")
    require(deltaHigh >= deltaLow, "RandRotation upper must be >= lower.")
    require(deltaLow >= 0, "RandRotation lower must be non-negative.")

    override def transformMat(feature: ImageFeature): Unit = {
      RandRotation.transform(feature.opencvMat(), feature.opencvMat(), RNG.uniform(deltaLow, deltaHigh))
    }
  }

  object RandRotation {
    def apply(deltaLow: Double, deltaHigh: Double): RandRotation = new RandRotation(deltaLow, deltaHigh)

    def transform(input: OpenCVMat, output: OpenCVMat, delta: Double): OpenCVMat = {
      if (delta >= 0&&delta < 1) {
        input.copyTo(output)
      }
      else if(delta >= 1&&delta < 2){
        var temp_image=input
        Core.transpose(input, temp_image)
        Core.flip(temp_image, output,1)
      }
      else if(delta >= 2&&delta < 3){
        var temp_image=input
        Core.transpose(input, temp_image)
        Core.flip(temp_image, output,0)
      }
      else{
        var temp_image=input
        Core.transpose(input, temp_image)
        Core.flip(temp_image, output,-1)
      }
      output
    }
  }


  /**
    * Adjust the image contrast
    * @param deltaLow contrast parameter low bound
    * @param deltaHigh contrast parameter high bound
    */
  class ContrastLog(deltaLow: Double, deltaHigh: Double)
    extends FeatureTransformer {

    require(deltaHigh >= deltaLow, "contrast upper must be >= lower.")
    require(deltaLow >= 0, "contrast lower must be non-negative.")
    override def transformMat(feature: ImageFeature): Unit = {
      ContrastLog.transform(feature.opencvMat(), feature.opencvMat(), RNG.uniform(deltaLow, deltaHigh))
    }
  }
  import org.opencv.imgproc.Imgproc
  object ContrastLog {
    def apply(deltaLow: Double, deltaHigh: Double): ContrastLog = new ContrastLog(deltaLow, deltaHigh)

    def transform(input: OpenCVMat, output: OpenCVMat, delta: Double): OpenCVMat = {
      if (Math.abs(delta - 1) > 1e-3) {

        // Convert to HSV colorspae
        Imgproc.cvtColor(input, output, Imgproc.COLOR_BGR2HSV)

        // Split the image to 3 channels.
        val channels = new util.ArrayList[Mat]()
        Core.split(output, channels)

        // Adjust the contrast.
        val temp_image=channels.get(2)
        Core.normalize(channels.get(2), channels.get(2))
        temp_image.convertTo(temp_image, -1, delta, 1)
        Core.log(temp_image, temp_image)
        temp_image.convertTo(temp_image, -1, 255/Math.log(1+delta), 0)


        Core.merge(channels, output)
        (0 until channels.size()).foreach(channels.get(_).release())
        // Back to BGR colorspace.
        Imgproc.cvtColor(output, output, Imgproc.COLOR_HSV2BGR)

      } else {
        if (input != output) input.copyTo(output)
      }
      output
    }
  }


  def valD(rdd: ImageFrame, sc: SparkContext, imageSize: Int, batchSize: Int, portion: Double, deltaHue: Double, deltaContrast: Double, deltaRotation: Double)
  : DataSet[MiniBatch[Float]] = {

    //Nonlinear logarithmic function
    //val deltaContrastLog=Math.log(deltaContrast)

    SeqFileFolder2.imageFrameToImageFeatureDataset(rdd).transform(
      ImageFeature2Batch(
        width = imageSize,
        height = imageSize,
        batchSize = batchSize,
        transformer = PixelBytesToMat() ->
          RandomResize(256, 256) ->
          RandomCropper(224, 224, true, CropRandom) ->

          //RandRotation(0.0, deltaRotation) ->

          Hue(deltaHue, deltaHue) ->
          ContrastLog(deltaContrast, deltaContrast) ->

          //1/255
          ChannelScaledNormalizer(104, 117, 123, 0.0078125) ->
          MatToTensor[Float](), portion, toRGB = false
      )

    )

  }

  def trainD(rdd: ImageFrame, sc: SparkContext, imageSize: Int, batchSize: Int, deltaHue: Double, deltaContrast: Double, deltaRotation: Double)
  : DataSet[MiniBatch[Float]] = {

    //Nonlinear logarithmic function
    //val deltaContrastLog=Math.log(deltaContrast)

    SeqFileFolder2.imageFrameToImageFeatureDataset(rdd).transform(
      MTImageFeatureToBatch(
        width = imageSize,
        height = imageSize,
        batchSize = batchSize,
        transformer = PixelBytesToMat() ->
          RandomAlterAspect() ->
          RandomCropper(224, 224, true, CropRandom) ->

          RandRotation(0.0, deltaRotation) ->

          Hue(deltaHue, deltaHue) ->
          ContrastLog(deltaContrast, deltaContrast) ->

          ChannelScaledNormalizer(104, 117, 123, 0.0078125) ->
          MatToTensor[Float](), toRGB = false
      )
    )
  }
}

