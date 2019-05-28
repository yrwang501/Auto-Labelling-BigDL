
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

import org.opencv.core.{Core, CvType, Mat, Point, Size, Rect}
import org.opencv.imgproc.Imgproc

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
    * Adjust the image RandRotation
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
    * Adjust the image RandRotation
    * @param rotationTimes Rotation times ineed
    */
  class RandAngleRotation(rotationTimes: Int)//0-10
    extends FeatureTransformer {

/*    require(deltaHigh <= 4&& deltaLow>=0, "RandRotation lower and upper must be in limit.")
    require(deltaHigh >= deltaLow, "RandRotation upper must be >= lower.")
    require(deltaLow >= 0, "RandRotation lower must be non-negative.")*/

    override def transformMat(feature: ImageFeature): Unit = {
      RandAngleRotation.transform(feature.opencvMat(), feature.opencvMat(), rotationTimes)
    }
  }

  object RandAngleRotation {
    def apply(rotationTimes: Int): RandAngleRotation = new RandAngleRotation(rotationTimes)

    def transform(input: OpenCVMat, output: OpenCVMat, times: Double): OpenCVMat = {
      val angle :Int=RNG.uniform(0, times).toInt*(360/times).toInt //RNG.uniform(0, times)

      //val radian :Float= (angle.toFloat /180.0*math.Pi).toFloat

      /*
      //fill image
      val maxBorder :Int=(math.max(input.cols, output.rows)* 1.414 ).toInt //sqrt(2)*max
      val dx :Int= (maxBorder - input.cols)/2
      val dy :Int= (maxBorder - input.rows)/2
      Core.copyMakeBorder(input, output, dy, dy, dx, dx, Core.BORDER_CONSTANT)*/

      //rotation
      if (angle==0) {
        input.copyTo(output)
      }
      else if(angle==90){
        var temp_image=input
        Core.transpose(input, temp_image)
        Core.flip(temp_image, output,1)
      }
      else if(angle==180){
        var temp_image=input
        Core.transpose(input, temp_image)
        Core.flip(temp_image, output,0)
      }
      else if(angle==270){
        var temp_image=input
        Core.transpose(input, temp_image)
        Core.flip(temp_image, output,-1)
      }
      else{

        val center=new Point( (output.cols/2).toDouble , (output.rows/2).toDouble)
        val affine_matrix = Imgproc.getRotationMatrix2D( center, angle, 1.0 )
        Imgproc.warpAffine(input, output, affine_matrix, output.size())
      }


      /*
      //Calculates the largest rectangle of the image after the image is rotated
      val sinVal :Float= math.abs(math.sin(radian)).toFloat
      val cosVal :Float= math.abs(math.cos(radian)).toFloat
      val targetSize =new Size((input.cols * cosVal +input.rows * sinVal).toDouble,(input.cols * sinVal + input.rows * cosVal).toDouble )

      //Cut off the extra border
      val x :Int= (output.cols - targetSize.width).toInt / 2
      val y :Int= (output.rows - targetSize.height).toInt / 2
      val rect=new Rect(x, y, targetSize.width.toInt, targetSize.height.toInt)
      val output_mat = new Mat(output,rect)

      val output_ = new OpenCVMat(output_mat)
      output_.copyTo(output)*/
      output
    }
  }


  /**
    * Adjust the image contrast using Log function
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

  object ContrastLog {
    def apply(deltaLow: Double, deltaHigh: Double): ContrastLog = new ContrastLog(deltaLow, deltaHigh)

    def transform(input: OpenCVMat, output: OpenCVMat, delta: Double): OpenCVMat = {
      if (Math.abs(delta - 1) > 1e-3) {
        input.convertTo(output,CvType.CV_32F)
        output.convertTo(output, -1, delta/255.0, 1)
        Core.log(output, output)
        output.convertTo(output, -1, Math.log(1+delta), 0)
        Core.normalize(output,output,0,255, Core.NORM_MINMAX)
        output.convertTo(output,CvType.CV_8UC3)
      } else {
        if (input != output) input.copyTo(output)
      }
      output
    }
  }

  /**
   * Adjust the image contrast using gamma function
   * @param deltaLow contrast parameter low bound
   * @param deltaHigh contrast parameter high bound
   */
  class ContrastGamma(deltaLow: Double, deltaHigh: Double)
    extends FeatureTransformer {

    require(deltaHigh >= deltaLow, "contrast upper must be >= lower.")
    require(deltaLow >= 0, "contrast lower must be non-negative.")
    override def transformMat(feature: ImageFeature): Unit = {
      ContrastGamma.transform(feature.opencvMat(), feature.opencvMat(), RNG.uniform(deltaLow, deltaHigh))
    }
  }

  object ContrastGamma {
    def apply(deltaLow: Double, deltaHigh: Double): ContrastGamma = new ContrastGamma(deltaLow, deltaHigh)

    def transform(input: OpenCVMat, output: OpenCVMat, delta: Double): OpenCVMat = {
      if (Math.abs(delta - 1) > 1e-3) {
        input.convertTo(output,CvType.CV_32F)
        output.convertTo(output, -1, 1.0/255, 0)
        Core.pow(output, delta, output)
        output.convertTo(output, -1, 255.0, 0)
        output.convertTo(output,CvType.CV_8UC3)

      } else {
        if (input != output) input.copyTo(output)
      }
      output
    }
  }

  def valD(rdd: ImageFrame, sc: SparkContext, imageSize: Int, batchSize: Int, portion: Double, deltaHue: Double, deltaContrast: Double)
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
          ContrastGamma(deltaContrast, deltaContrast) ->

          //1/255
          ChannelScaledNormalizer(104, 117, 123, 0.0078125) ->
          MatToTensor[Float](), portion, toRGB = false
      )

    )

  }

  def trainD(rdd: ImageFrame, sc: SparkContext, imageSize: Int, batchSize: Int, deltaHue: Double, deltaContrast: Double, deltaRotationTimes: Int)
  : DataSet[MiniBatch[Float]] = {

    //Nonlinear logarithmic function
    //val deltaContrastLog=Math.log(deltaContrast)

    SeqFileFolder2.imageFrameToImageFeatureDataset(rdd).transform(
      MTImageFeatureToBatch(
        width = imageSize,
        height = imageSize,
        batchSize = batchSize,
        transformer = PixelBytesToMat() ->
          RandAngleRotation(deltaRotationTimes) ->

          RandomAlterAspect() ->
          RandomCropper(224, 224, true, CropRandom) ->

          //RandRotation(0.0, deltaRotation) ->

          Hue(deltaHue, deltaHue) ->
          ContrastGamma(deltaContrast, deltaContrast) ->

          //Contrast

          ChannelScaledNormalizer(104, 117, 123, 0.0078125) ->
          MatToTensor[Float](), toRGB = false
      )
    )
  }
}

