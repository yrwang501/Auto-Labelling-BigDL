package com.intel.analytics.bigdl.dataset

import java.awt.image.{BufferedImage, DataBufferByte}
import java.nio.ByteBuffer
import java.nio.file.{Files, Path, Paths}

import com.intel.analytics.bigdl.DataSet
import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.dataset.image.{BGRImage, LabeledBGRImage, LocalImageFiles, LocalLabeledImagePath}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.{DistributedImageFrame, ImageFeature, ImageFrame, LocalImageFrame}
import com.intel.analytics.bigdl.utils.{Engine, T}
import javax.imageio.ImageIO
import org.apache.hadoop.io.Text
import org.apache.log4j.Logger
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import scala.reflect.{ClassTag, classTag}
import scala.util.Random



/**
  * Common used DataSet builder.
  */
object DataSet  {
  val logger = Logger.getLogger(getClass)

  /**
    * Wrap an array as a DataSet.
    */
  def array[T](data: Array[T]): LocalArrayDataSet[T] = {
    new LocalArrayDataSet[T](data)
  }

  /**
    * Wrap an array as a distributed DataSet.
    * @param localData
    * @param sc
    * @tparam T
    * @return
    */
  def array[T: ClassTag](localData: Array[T], sc: SparkContext): DistributedDataSet[T] = {
    val nodeNumber = Engine.nodeNumber()
    new CachedDistriDataSet[T](
      sc.parallelize(localData, nodeNumber)
        // Keep this line, or the array will be send to worker every time
        .coalesce(nodeNumber, true)
        .mapPartitions(iter => {
          Iterator.single(iter.toArray)
        }).setName("cached dataset")
        .cache()
    )
  }

  /**
    * Wrap a RDD as a DataSet.
    * @param data
    * @tparam T
    * @return
    */
  def rdd[T: ClassTag](data: RDD[T]): DistributedDataSet[T] = {
    val nodeNumber = Engine.nodeNumber()
    new CachedDistriDataSet[T](
      data.coalesce(nodeNumber, true)
        .mapPartitions(iter => {
          Iterator.single(iter.toArray)
        }).setName("cached dataset")
        .cache()
    )
  }

  def imageFrame(imageFrame: ImageFrame): DataSet[ImageFeature] = {
    imageFrame match {
      case distributedImageFrame: DistributedImageFrame =>
        rdd[ImageFeature](distributedImageFrame.rdd)
      case localImageFrame: LocalImageFrame =>
        array(localImageFrame.array)
    }
  }

  /**
    * Wrap a RDD as a DataSet.
    * @param data
    * @tparam T
    * @return
    */
  private[bigdl] def sortRDD[T: ClassTag](data: RDD[T], isInOrder: Boolean = false,
                                          groupSize: Int = 1): DistributedDataSet[T] = {
    val nodeNumber = Engine.nodeNumber()
    new CachedDistriDataSet[T](
      data.coalesce(nodeNumber, true)
        .mapPartitions(iter => {
          Iterator.single(sortData(iter.toArray, isInOrder))
        }).setName("cached dataset")
        .cache(),
      isInOrder,
      groupSize
    )
  }

  /**
    * sort data from small to big, only support Sample data type.
    * @param data original data
    * @param isInOrder whether to sort data by ascending order
    * @return
    */
  def sortData[T: ClassTag](data: Array[T], isInOrder: Boolean): Array[T] = {
    if (isInOrder) {
      require(classTag[T] == classTag[Sample[_]],
        "DataSet.sortData: Only support sort for sample input")
      data.sortBy(a => a.asInstanceOf[Sample[_]].featureLength(0))
    } else {
      data
    }
  }

  /**
    * Generate a DataSet from a local image folder. The image folder should have two levels. The
    * first level is class folders, and the second level is images. All images belong to a same
    * class should be put into the same class folder. So each image in the path is labeled by the
    * folder it belongs.
    */
  object ImageFolder {
    /**
      * Extract all image paths into a Local DataSet. The paths are all labeled. When the image
      * files are too large(e.g. ImageNet2012 data set), you'd better readd all paths instead of
      * image files themselves.
      * @param path
      * @return
      */
    def paths(path: Path): LocalDataSet[LocalLabeledImagePath] = {
      val buffer = LocalImageFiles.readPaths(path)
      new LocalArrayDataSet[LocalLabeledImagePath](buffer)
    }

    /**
      * Extract all images under the given path into a Local DataSet. The images are all labeled.
      * @param path
      * @param scaleTo
      * @return
      */
    def images(path: Path, scaleTo: Int): DataSet[LabeledBGRImage] = {
      val paths = LocalImageFiles.readPaths(path)
      val total = paths.length
      var count = 1
      val buffer = paths.map(imageFile => {
        if (total < 100 || count % (total / 100) == 0 || count == total) {
          logger.info(s"Cache image $count/$total(${count * 100 / total}%)")
        }
        count += 1

        val bufferBGR = new LabeledBGRImage()
        bufferBGR.copy(BGRImage.readImage(imageFile.path, scaleTo), 255f)
          .setLabel(imageFile.label)
      })

      new LocalArrayDataSet[LabeledBGRImage](buffer)
    }

    /**
      * Extract all images under the given path into a Distributed DataSet. The images are all
      * labeled.
      * @param path
      * @param sc
      * @param scaleTo
      * @return
      */
    def images(path: Path, sc: SparkContext, scaleTo: Int)
    : DataSet[LabeledBGRImage] = {
      val paths = LocalImageFiles.readPaths(path)
      val buffer: Array[LabeledBGRImage] = {
        paths.map(imageFile => {
          val bufferBGR = new LabeledBGRImage()
          bufferBGR.copy(BGRImage.readImage(imageFile.path, scaleTo), 255f)
            .setLabel(imageFile.label)
        })
      }
      array(buffer, sc)
    }
  }

  /**
    * Create a DataSet from a Hadoop sequence file folder.
    */
  object SeqFileFolder2 {
    val logger = Logger.getLogger(getClass)

    /**
      * Extract all hadoop sequence file paths from a local file folder.
      * @param path
      * @param totalSize
      * @return
      */
    def paths(path: Path, totalSize: Long): LocalDataSet[LocalSeqFilePath] = {
      logger.info(s"Read sequence files folder $path")
      val buffer: Array[LocalSeqFilePath] = findFiles(path)
      logger.info(s"Find ${buffer.length} sequence files")
      require(buffer.length > 0, s"Can't find any sequence files under $path")
      new LocalArrayDataSet[LocalSeqFilePath](buffer) {
        override def size(): Long = {
          totalSize
        }
      }
    }

    /**
      * get label from text of sequence file,
      * @param data text of sequence file, this text can split into parts by "\n"
      * @return
      */
    def readLabel(data: Text): String = {
      val dataArr = data.toString.split("\n")
      if (dataArr.length == 1) {
        dataArr(0)
      } else {
        dataArr(1)
      }
    }

    /**
      * get name from text of sequence file,
      * @param data text of sequence file, this text can split into parts by "\n"
      * @return
      */
    def readName(data: Text): String = {
      val dataArr = data.toString.split("\n")
      require(dataArr.length >= 2, "key in seq file only contains label, no name")
      dataArr(0)
    }

    /**
      * Extract hadoop sequence files from an HDFS path
      * @param url
      * @param sc
      * @param classNum
      * @return
      */
    def files(url: String, sc: SparkContext, classNum: Int): DistributedDataSet[ByteRecord] = {
      val nodeNumber = Engine.nodeNumber()
      val coreNumber = Engine.coreNumber()
      val rawData = sc.sequenceFile(url, classOf[Text], classOf[Text],
        nodeNumber * coreNumber).map(image => {
        ByteRecord(image._2.copyBytes(), readLabel(image._1).toFloat)
      }).filter(_.label <= classNum)

      rdd[ByteRecord](rawData)
    }

    /**
      * Extract hadoop sequence files from an HDFS path as RDD
      * @param url sequence files folder path
      * @param sc spark context
      * @param classNum class number of data
      * @param partitionNum partition number, default: Engine.nodeNumber() * Engine.coreNumber()
      * @return
      */
    private[bigdl] def filesToRdd(url: String, sc: SparkContext,
                                  classNum: Int, partitionNum: Option[Int] = None): RDD[ByteRecord] = {
      val num = partitionNum.getOrElse(Engine.nodeNumber() * Engine.coreNumber())
      val rawData = sc.sequenceFile(url, classOf[Text], classOf[Text], num).map(image => {
        ByteRecord(image._2.copyBytes(), readLabel(image._1).toFloat)
      }).filter(_.label <= classNum)
      rawData.coalesce(num, true)
    }

    /**
      * Extract hadoop sequence files from an HDFS path as ImageFrame
      * @param url sequence files folder path
      * @param sc spark context
      * @param classNum class number of data
      * @param partitionNum partition number, default: Engine.nodeNumber() * Engine.coreNumber()
      * @return
      */
    private[bigdl] def filesToImageFrame(url: String, sc: SparkContext,
                                         classNum: Int, partitionNum: Option[Int] = None): ImageFrame = {
      val num = partitionNum.getOrElse(Engine.nodeNumber() * Engine.coreNumber())
      val rawData = sc.sequenceFile(url, classOf[Text], classOf[Text], num).map(image => {
        val rawBytes = image._2.copyBytes()
        val label = Tensor[Float](T(readLabel(image._1).toFloat))
        val imgBuffer = ByteBuffer.wrap(rawBytes)
        val width = imgBuffer.getInt
        val height = imgBuffer.getInt
        val bytes = new Array[Byte](3 * width * height)
        System.arraycopy(imgBuffer.array(), 8, bytes, 0, bytes.length)
        val imf = ImageFeature(bytes, label)
        imf(ImageFeature.originalSize) = (height, width, 3)
        imf
      }).filter(_[Tensor[Float]](ImageFeature.label).valueAt(1) <= classNum)
      ImageFrame.rdd(rawData)
    }

    private[bigdl] def filesToImageFeatureDataset(url: String, sc: SparkContext,
                                                  classNum: Int, partitionNum: Option[Int] = None): DistributedDataSet[ImageFeature] = {
      rdd[ImageFeature](filesToImageFrame(url, sc, classNum, partitionNum).toDistributed().rdd)
    }

    private[bigdl] def findFiles(path: Path): Array[LocalSeqFilePath] = {
      val directoryStream = Files.newDirectoryStream(path)
      import scala.collection.JavaConverters._
      directoryStream.asScala.map(_.toAbsolutePath.toString)
        .filter(_.endsWith(".seq")).toArray.sortWith(_ < _).map(p => LocalSeqFilePath(Paths.get(p)))
    }

    /**
      * SuSuSuSu
      * @param path
      * @return
      */
//    private[bigdl] def rddToImageFrame(inRdd: RDD[String], sc: SparkContext,
//                                       classNum: Int, partitionNum: Option[Int] = None): ImageFrame = {
//      val num = partitionNum.getOrElse(Engine.nodeNumber() * Engine.coreNumber())
//      val rawData = inRdd.map(row => {
//        //val rawString = row.drop(0).dropRight(1)
//        val rawString = row
//        val base64String: String = rawString.split(";")(4)
//        val rawBytes: Array[Byte] = javax.xml.bind.DatatypeConverter
//          .parseBase64Binary(base64String.map{case '-' => '+'; case '_' => '/'; case c => c })
//
//        import java.io.ByteArrayInputStream
//        val pixelArr = new ByteArrayInputStream(rawBytes)
//        val image = ImageIO.read(pixelArr)
//
//        val bytes: Array[Byte] = image.getRaster.getDataBuffer.asInstanceOf[DataBufferByte].getData
//        val label = Tensor[Float](T(rawString.split(";")(3).toFloat + 1))
//
//        val imf = ImageFeature(bytes, label)
//        imf(ImageFeature.originalSize) = (512, 512, 3)
//        imf
//      }).filter(_[Tensor[Float]](ImageFeature.label).valueAt(1) <= classNum)
//      ImageFrame.rdd(rawData)
//    }

    private[bigdl] def rddToImageFrame(inRdd: RDD[String], sc: SparkContext,
                                       classNum: Int, partitionNum: Option[Int] = None, cropWidth: Int = 150, cropHeight: Int = 150): ImageFrame = {
      val num = partitionNum.getOrElse(Engine.nodeNumber() * Engine.coreNumber())
      val rawData = inRdd.map(row => {
        //val rawString = row.drop(0).dropRight(1)
        val rawString = row
        val base64String: String = rawString.split(";")(3)
        val rawBytes: Array[Byte] = javax.xml.bind.DatatypeConverter
          .parseBase64Binary(base64String.map{case '-' => '+'; case '_' => '/'; case c => c })

        import java.io.ByteArrayInputStream
        val pixelArr = new ByteArrayInputStream(rawBytes)
        val image = ImageIO.read(pixelArr)

        val label = Tensor[Float](T(rawString.split(";")(1).toFloat + 1))
        val imf = new ImageFeature()

        val offset = row.split(";")(2)

        def getCoordinates(offset: String, width: Int, height: Int): (Int, Int) = {
          if (offset == "#") {
            // when meeting one negative image, randomly cop a bounding bos of 150 * 150
            val x = Random.nextInt(512 - width)
            val y = Random.nextInt(512 - height)
            (x, y)
          }
          else {
            val xStr = offset.split(",")(0)
            val yStr = offset.split(",")(1)
            (xStr.toInt, yStr.toInt)
          }
        }

        val (x, y) = getCoordinates(offset, cropWidth, cropHeight)
        val subImage = new BufferedImage(cropWidth, cropHeight, 5)
        val graphics = subImage.getGraphics
        graphics.drawImage(image.getSubimage(x, y, cropWidth, cropHeight), 0, 0, null)
        graphics.dispose()
        val croppedBytes: Array[Byte] = subImage.getRaster.getDataBuffer.asInstanceOf[DataBufferByte].getData()
        imf(ImageFeature.bytes) = croppedBytes
        imf(ImageFeature.label) = label
        imf(ImageFeature.originalSize) = (cropWidth, cropHeight, 3)
        imf
      }).filter(_[Tensor[Float]](ImageFeature.label).valueAt(1) <= classNum)
      ImageFrame.rdd(rawData)
    }

    private[bigdl] def rddToImageFeatureDataset(inRdd: RDD[String], sc: SparkContext,
                                                classNum: Int, partitionNum: Option[Int] = None): DistributedDataSet[ImageFeature] = {
      rdd[ImageFeature](rddToImageFrame(inRdd, sc, classNum, partitionNum).toDistributed().rdd)
    }

    private[bigdl] def imageFrameToImageFeatureDataset(inRdd: ImageFrame): DistributedDataSet[ImageFeature] = {
      rdd[ImageFeature](inRdd.toDistributed().rdd)
    }
  }

}
