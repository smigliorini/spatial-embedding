/*
 * Copyright 2023 University of California, Riverside
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package edu.ucr.cs.bdlab

import org.apache.spark.SparkConf

object SummarizeRealData {
  def main(args: Array[String]): Unit = {
    import org.apache.hadoop.fs.Path
    import org.apache.spark.SparkContext
    import org.apache.spark.sql.SparkSession

    import java.io.{FileOutputStream, PrintStream}
    import java.util.concurrent.TimeoutException
    import scala.concurrent.Future

    // Create the Spark context
    val conf = new SparkConf
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.setAppName("SummarizeRealData")

    // Set Spark master to local if not already set
    if (!conf.contains("spark.master"))
      conf.setMaster("local[*]")
    val spark: SparkSession = SparkSession.builder().config(conf).getOrCreate()
    val sc: SparkContext = spark.sparkContext
    import edu.ucr.cs.bdlab.beast._

    import scala.concurrent.Await
    import scala.concurrent.ExecutionContext.Implicits.global
    import scala.concurrent.duration.Duration

    val inputDir = new Path("lakes_parks")
    val localSummariesDir = new Path("lakes_parks_summaries")
    val globalSummaryOutput = new PrintStream(new FileOutputStream("lakes_parks_global_summaries.csv"))
    globalSummaryOutput.println(Array("dataset", "distribution", "x1", "y1", "x2", "y2", "num_features", "size", "num_points",
      "avg_area", "avg_side_length_0", "avg_side_length_1", "E0", "E2").mkString(","))
    val fileSystem = inputDir.getFileSystem(sc.hadoopConfiguration)

    case class GlobalSummary(dataset: String, distribution: String, x1: Double,	y1: Double,	x2: Double,	y2: Double,	num_features: Long,
                             size: Long, num_points: Long, avg_area: Double, avg_side_length: Array[Double],
                             e0: Double, e2: Double) {
      def toSeq: Seq[Any] = Seq(dataset, distribution, x1, y1, x2, y2, num_features, size,
        num_points, avg_area, avg_side_length(0), avg_side_length(1), e0, e2)
    }

    val inputFiles = new collection.mutable.ArrayBuffer[String]()
    inputFiles ++= fileSystem.listStatus(inputDir).map(_.getPath.getName)
    val filesBeingProcessed = new collection.mutable.ArrayBuffer[Future[GlobalSummary]]()

    val parallelism = 32

    while (inputFiles.nonEmpty || filesBeingProcessed.nonEmpty) {
      // Remove all ones that are done
      var i = 0
      while (i < filesBeingProcessed.length) {
        try {
          val result: GlobalSummary = Await.result(filesBeingProcessed(i), Duration.fromNanos(1E9))
          globalSummaryOutput.println(result.toSeq.mkString(","))
          filesBeingProcessed.remove(i)
        } catch {
          case _: TimeoutException | _: InterruptedException => i += 1
        }
      }
      while (inputFiles.nonEmpty && filesBeingProcessed.length < parallelism) {
        val filename = inputFiles.remove(inputFiles.length - 1)
        println(s"Processing ${filename}")
        filesBeingProcessed.append(Future[GlobalSummary] {
          try {
            val summary = sc.readWKTFile(new Path(inputDir, filename).toString, 0, skipHeader = true).summary
            GlobalSummary(filename, "real", summary.getMinCoord(0), summary.getMinCoord(1), summary.getMaxCoord(0), summary.getMaxCoord(1),
              summary.numFeatures, summary.size, summary.numPoints, summary.averageArea,
              Array(summary.averageSideLength(0), summary.averageSideLength(1)), 0, 0)
          } catch {
            case e: Exception => e.printStackTrace(); throw e
          }

        })
      }
    }

    globalSummaryOutput.close()
  }

}
