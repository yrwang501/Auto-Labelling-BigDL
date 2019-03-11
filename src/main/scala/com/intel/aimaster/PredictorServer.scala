package com.intel.aimaster


import java.util.concurrent.atomic.AtomicInteger

import scala.io.StdIn
import com.intel.analytics.bigdl.models.resnet.test
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

import scala.concurrent.{Await, Future}
import org.http4s.server.blaze._
object PredictorServer extends  StreamApp[IO]{
  override def stream(args: List[String], requestShutdown: IO[Unit]): Stream[IO, ExitCode] = {
    val progress = new AtomicInteger()
    var modelInit = Future{
      test.init(args(0), args(1),
        args(2), args(3),
        Some(args(4)))
    }
    println(args)
    @volatile var isRunning = false
    val helloWorldService = HttpService[IO] {
      case GET -> Root / "hello" / name =>
        Ok(s"Hello, $name.")
      case GET -> Root / "predict" / startRow / length => {
        if(!modelInit.isCompleted || isRunning){
          Ok(s"""{"accepted":false,"start":$startRow,"len":$length}""")
            .map(_.withContentType(`Content-Type`(MediaType.`application/json`)))
        }
        else
        {
          isRunning=true
          progress.set(0)
          Future{
            test.doPredict(startRow.toInt, length.toInt)
            isRunning=false
          }
          Ok(s"""{"accepted":true,"start":$startRow,"len":$length}""")
            .map(_.withContentType(`Content-Type`(MediaType.`application/json`)))
        }
      }
      case GET -> Root / "reload" =>{
        val success = !isRunning & modelInit.isCompleted
        if(success)
          modelInit = Future{
            test.loadAndBcastModel(args(2))
          }
        Ok(s"""{"accepted":$success}""")
          .map(_.withContentType(`Content-Type`(MediaType `application/json`)))
      }
      case GET -> Root / "updateStatus" =>{
        progress.addAndGet(80)
        Ok(s"{}")
          .map(_.withContentType(`Content-Type`(MediaType `application/json`)))
      }
      case GET -> Root / "status" =>{
        val pval=progress.get()
        val total=test.numOfPredictions * 16
        val status = if(!modelInit.isCompleted){
          "loading"
        }
        else if (isRunning){
          "running"
        }
        else{
          "idle"
        }
        Ok(s"""{"status":"$status", "progress":$pval, "total":$total}""")
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

    BlazeBuilder[IO].bindHttp(13345, "0.0.0.0").mountService(CORS(helloWorldService, methodConfig), "/").serve
  }
}