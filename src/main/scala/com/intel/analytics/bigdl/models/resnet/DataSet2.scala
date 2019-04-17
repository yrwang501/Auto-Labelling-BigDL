
package com.intel.analytics.bigdl.models.resnet

import com.intel.analytics.bigdl.{DataSet, dataset}
import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.dataset.image._
import com.intel.analytics.bigdl.dataset.image.{HFlip => JHFlip}
import com.intel.analytics.bigdl.dataset.DataSet2.SeqFileFolder2
import com.intel.analytics.bigdl.transform.vision
import com.intel.analytics.bigdl.transform.vision.image._
import com.intel.analytics.bigdl.transform.vision.image.augmentation._
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

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

  def valD(rdd: ImageFrame, sc: SparkContext, imageSize: Int, batchSize: Int, portion: Double)
  : DataSet[MiniBatch[Float]] = {
    SeqFileFolder2.imageFrameToImageFeatureDataset(rdd).transform(
      ImageFeature2Batch(
        width = imageSize,
        height = imageSize,
        batchSize = batchSize,
        transformer = PixelBytesToMat() ->
          RandomResize(256, 256) ->
          RandomCropper(224, 224, true, CropRandom) ->
          ChannelScaledNormalizer(104, 117, 123, 0.0078125) ->
          MatToTensor[Float](), portion, toRGB = false
      )

    )

  }

  def trainD(rdd: ImageFrame, sc: SparkContext, imageSize: Int, batchSize: Int)
  : DataSet[MiniBatch[Float]] = {
    SeqFileFolder2.imageFrameToImageFeatureDataset(rdd).transform(
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
}

