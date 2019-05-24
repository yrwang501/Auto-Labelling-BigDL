package com.intel.aimaster


import java.util.concurrent.atomic.AtomicInteger

import scala.io.StdIn
import cats.effect._
import org.http4s._
import org.http4s.dsl.io._
import io.circe.Json

import scala.concurrent.duration._
import scala.concurrent.ExecutionContext.Implicits.global
import org.http4s.server.blaze._
import fs2.{Stream, StreamApp}
import fs2.StreamApp.ExitCode
import org.http4s.headers._
import org.http4s.server.middleware._

import scala.concurrent.{Await, Future, Promise}
import org.http4s.server.blaze._
object TrainServer extends  StreamApp[IO]{


  override def stream(args: List[String], requestShutdown: IO[Unit]): Stream[IO, ExitCode] = {

    //0=coresite, 1=hbasesit, 2=tablename, 3=startrow, 4=endrow, 5=batchsize, 6=testsize, 7=maxepoch
    val batchSize = args(5).toInt
    val validatePortition = args(6).toDouble
    val maxEpoch = args(7).toInt
    var modelInit= Future{
      Train.init(args(0), args(1), args(2))
      Train.loadDataset(args(3).toInt,args(4).toInt,batchSize,validatePortition)
    }
    println(args)
    val helloWorldService = HttpService[IO] {
      case GET -> Root / "hello" / name =>
        Ok(s"Hello, $name.")
      case req @ GET -> Root / "getmodel" =>

        val hue = req.params.get("hue")
        val contrast = req.params.get("contrast")
        val lr = req.params.get("lr")
        val rotations = req.params.get("rotations")
        val modelpath = req.params.get("modelpath")
        val epochs = req.params.get("epochs")
        if (hue.isEmpty || contrast.isEmpty || lr.isEmpty || rotations.isEmpty
          || modelpath.isEmpty || epochs.isEmpty
        ) {
          BadRequest(s"Invalid data: " + req.params)
        } else {
          if(!modelInit.isCompleted || Train.isTraining){
            Ok(s"""{"accepted":false,"hue":${hue.get.toDouble},"contrast":${contrast.get.toDouble}}""")
              .map(_.withContentType(`Content-Type`(MediaType.`application/json`)))
          }
          else {
            println(s"epochs=${epochs.get.toInt} hue=${hue.get.toDouble}"
              + s" contrast=${contrast.get.toDouble}, learningRate = ${lr.get.toDouble}, rotation = ${rotations.get.toDouble}")
            Future {
              Train.doTrain(batchSize, epochs.get.toInt, 1, 1,
                modelpath, validatePortition = validatePortition,
                deltaHue = hue.get.toDouble, deltaContrast = contrast.get.toDouble,
                learningRate = lr.get.toDouble, deltaRotation = rotations.get.toDouble)
            }
            Ok(s"""{"accepted":true,"hue":${hue.get.toDouble},"contrast":${contrast.get.toDouble}}""")
              .map(_.withContentType(`Content-Type`(MediaType.`application/json`)))
          }
        }
      case GET -> Root / "train" / startRow / stopRow / deltaHue / deltaContrast / learningRate / deltaRotation=> {
        if(!modelInit.isCompleted || Train.isTraining){
          Ok(s"""{"accepted":false,"start":$startRow,"len":$stopRow}""")
            .map(_.withContentType(`Content-Type`(MediaType.`application/json`)))
        }
        else
        {
          Future{
            Train.doTrain(batchSize, maxEpoch, startRow.toInt, stopRow.toInt,
              None, validatePortition = validatePortition,
              deltaHue = deltaHue.toDouble, deltaContrast = deltaContrast.toDouble,
              learningRate = learningRate.toDouble, deltaRotation = deltaRotation.toDouble)
          }
          Ok(s"""{"accepted":true,"start":$startRow,"len":$stopRow}""")
            .map(_.withContentType(`Content-Type`(MediaType.`application/json`)))
        }
      }
      case GET -> Root / "reload"   =>{
        val success = !Train.isTraining & modelInit.isCompleted
        if(success) {
          modelInit = Future {
            Train.init(args(0), args(1), args(2))
          }
        }
        Ok(s"""{"accepted":$success}""")
          .map(_.withContentType(`Content-Type`(MediaType `application/json`)))
      }
      case GET -> Root / "status" =>{
        val status = if(!modelInit.isCompleted){
          "loading"
        }
        else if (Train.isTraining){
          "running"
        }
        else{
          "idle"
        }
        val prog=Train.niter
        val acc=Train.accScore
        Ok(s"""{"status":"$status", "progress":$prog, "accuracy":$acc}""")
          .map(_.withContentType(`Content-Type`(MediaType.`application/json`)))
      }
      case GET -> Root / "stop" =>{
        Ok(s"""{"status":"ok"}""")
           .map(_.withContentType(`Content-Type`(MediaType.`application/json`)))
        sys.exit()
      }
      case GET -> Root / "cancel" =>{
        if(Train.isTraining){
          Train.needsAbort = true
        }
        Ok(s"""{"status":"ok"}""")
          .map(_.withContentType(`Content-Type`(MediaType.`application/json`)))
      }
    }

    val methodConfig = CORSConfig(
      anyOrigin = true,
      anyMethod = false,
      allowedMethods = Some(Set("GET", "POST")),
      allowCredentials = true,
      maxAge = 1.day.toSeconds,
      allowedHeaders = Some(Set("Content-Type"))
    )

    BlazeBuilder[IO].bindHttp(13346, "0.0.0.0").mountService(CORS(helloWorldService, methodConfig), "/").serve
  }
}
