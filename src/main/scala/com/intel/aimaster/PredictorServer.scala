package com.intel.aimaster


import java.util.concurrent.atomic.AtomicInteger

import scala.io.StdIn
import com.intel.analytics.bigdl.models.resnet.test
import cats.effect._
import org.http4s._
import org.http4s.dsl.io._

import scala.concurrent.ExecutionContext.Implicits.global
import org.http4s.server.blaze._
import fs2.{Stream, StreamApp}
import fs2.StreamApp.ExitCode

import scala.concurrent.{Await, Future}
import org.http4s.server.blaze._
object PredictorServer extends  StreamApp[IO]{
  /*def main(args: Array[String]) {
    import com.typesafe.config.ConfigFactory

    implicit val system = ActorSystem("my-system", ConfigFactory.load(customConf))
    implicit val materializer = ActorMaterializer()
    // needed for the future flatMap/onComplete in the end
    implicit val executionContext = system.dispatcher
    test.init("core-site.xml", "hbase-site.xml",
      "hdfs://ai-master-bigdl-0.sh.intel.com:8020/model_new_helper_API_10.obj", "kfb_512_100_test",
      Some("http://chengguosu.sh.intel.com:13345/status"))
    var progress = 0
    val route =
      pathPrefix("predict" / IntNumber / IntNumber) { (startRow, length) =>
        get {
          progress = 0
          //test.doPredict(startRow, length)
          complete(HttpEntity(ContentTypes.`application/json`, s"{start:${startRow},len:${length}}"))
        }
      } ~
        path("updateStatus") {
          get {
            progress += 10
            complete(HttpEntity(ContentTypes.`application/json`, s"{}"))
          }
        } ~
        path("status") {
          get {
            complete(HttpEntity(ContentTypes.`application/json`, s"{progress:${progress}}"))
          }
        }

    val bindingFuture = Http().bindAndHandle(route, "0.0.0.0", 13345)

    println(s"Server online at http://localhost:13345/\nPress RETURN to stop...")
    StdIn.readLine() // let it run until user presses return
    bindingFuture
      .flatMap(_.unbind()) // trigger unbinding from the port
      .onComplete(_ => system.terminate()) // and shutdown when done
  }*/

  override def stream(args: List[String], requestShutdown: IO[Unit]): Stream[IO, ExitCode] = {
    var progress = new AtomicInteger()
    test.init("core-site.xml", "hbase-site.xml",
      "hdfs://ai-master-bigdl-0.sh.intel.com:8020/model_new_helper_API_10.obj", "kfb_512_100_test",
      Some("http://chengguosu.sh.intel.com:13345/updateStatus"))
    val helloWorldService = HttpService[IO] {
      case GET -> Root / "hello" / name =>
        Ok(s"Hello, $name.")
      case GET -> Root / "predict" / startRow / length => {
        progress.set(0)

        Future{
          test.doPredict(startRow.toInt, length.toInt)
        }
        Ok(s"{start:${startRow},len:${length}}")
      }
      case GET -> Root / "updateStatus" =>{
        progress.addAndGet(10)
        Ok(s"{}")
      }
      case GET -> Root / "status" =>{
        val pval=progress.get()
        Ok(s"{progress:${pval}}")
      }

    }
    BlazeBuilder[IO].bindHttp(13345, "0.0.0.0").mountService(helloWorldService, "/").serve
  }
}