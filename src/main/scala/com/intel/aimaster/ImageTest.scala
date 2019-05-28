package com.intel.aimaster

import com.intel.analytics.bigdl.models.resnet.ImageNetDataSet2.{ContrastGamma, RandAngleRotation}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.{ImageFeature, PixelBytesToMat}
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.bigdl.utils.T
import java.awt.Image
import org.opencv.core.Mat
import java.awt.image.BufferedImage
import java.awt.image.DataBufferByte
import javax.swing.ImageIcon
import javax.swing.JFrame
import javax.swing.JLabel
import java.awt.FlowLayout
import java.io.{ByteArrayInputStream, FileInputStream}
import javax.imageio.ImageIO
import org.opencv.imgcodecs.Imgcodecs
object ImageTest {


  def Mat2BufferedImage(m: Mat): BufferedImage = { // Fastest code
// output can be assigned either to a BufferedImage or to an Image
    var `type` = BufferedImage.TYPE_BYTE_GRAY
    if (m.channels > 1) `type` = BufferedImage.TYPE_3BYTE_BGR
    val bufferSize = m.channels * m.cols * m.rows
    val b = new Array[Byte](bufferSize)
    m.get(0, 0, b)// get all the pixels

    val image = new BufferedImage(m.cols, m.rows, `type`)
    val targetPixels = image.getRaster.getDataBuffer.asInstanceOf[DataBufferByte]
      .getData
    System.arraycopy(b, 0, targetPixels, 0, b.length)
    image
  }



  def displayImage(img2: Image, closeit: Boolean): Unit = { //BufferedImage img=ImageIO.read(new File
    // ("/HelloOpenCV/lena.png"));
    val icon = new ImageIcon(img2)
    val frame = new JFrame
    frame.setLayout(new FlowLayout)
    println(s"Disp: ${img2.getHeight(null)}, ${img2.getWidth(null)}")
    frame.setSize(img2.getWidth(null) + 50, img2.getHeight(null) + 50)
    val lbl = new JLabel
    lbl.setIcon(icon)
    frame.add(lbl)
    frame.setVisible(true)
    if(closeit) frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE)
  }

  def displayImage(img2: Mat, closeit: Boolean): Unit ={
    displayImage(Mat2BufferedImage(img2), closeit)
  }

  def main(argv: Array[String]) :Unit = {
    val pixelArr = new FileInputStream("/home/menooker/Downloads/bird.jpg")
    val image = ImageIO.read(pixelArr)
    println(s"${image.getHeight()}, ${image.getWidth()}")
    displayImage(image,false)
    while (true){
      val input = scala.io.StdIn.readLine()
      if (input.equals("q")) System.exit(0)
      val tranf = input.toDouble
      val label = Tensor[Float](T(1.0))
      val imf = new ImageFeature()
      val rawdata = image.getRaster.getDataBuffer.asInstanceOf[DataBufferByte].getData()
      imf(ImageFeature.bytes) = rawdata
      imf(ImageFeature.originalSize) = (image.getHeight, image.getWidth, 3)

      val origin =  PixelBytesToMat().transform(imf)
      val timg = RandAngleRotation(tranf.toInt).transform(origin).opencvMat()
      displayImage(timg,false)
    }

  }

}
