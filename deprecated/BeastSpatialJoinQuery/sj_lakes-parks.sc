import edu.ucr.cs.bdlab.beast.util.FileUtil
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
val conf = new SparkConf()
val spark: SparkSession = null
val sc: SparkContext = null
import edu.ucr.cs.bdlab.beast._

// Start copying from here
import java.io.{File, FileOutputStream, PrintStream}
import java.util.concurrent.TimeoutException
import scala.collection.mutable.ArrayBuffer
import edu.ucr.cs.bdlab.beast.cg.SpatialJoinAlgorithms.ESJPredicate
import edu.ucr.cs.bdlab.beast.indexing.RSGrovePartitioner
import edu.ucr.cs.bdlab.beast.operations.SpatialJoin
import scala.concurrent.{Await, Future}
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration._
import java.io.{BufferedReader, FileReader}

val scale = 1

val pairs = spark.read.option("delimiter", ",").option("header", true).csv("realDataset_jn_pairs_rot.csv").select("dataset1", "dataset2").collect
val resultFileName: String = "sj_real_results_revision.csv"
var existingResults: Array[(String, String)] = Array()
val outputResults: PrintStream = if (new File(resultFileName).exists()) {
  val file = new BufferedReader(new FileReader(resultFileName))
  var line: String = null
  do {
    line = file.readLine()
    if (line != null) {
      val split = line.trim.split(",")
      existingResults +:= ((split(0), split(1)))
    }
  } while (line != null)
  file.close()
  new PrintStream(new FileOutputStream(resultFileName, true))
} else {
  val ps = new PrintStream(new FileOutputStream(resultFileName))
  ps.println("dataset1,datasets2,cardinality1,cardinality2,djresultsize,selectivity,djmbrTests")
  ps
}
val maxParallelism = 32
try {
  val activeJobs = new ArrayBuffer[Future[Unit]]()
  for (pair <- pairs) {
    while (activeJobs.size >= maxParallelism) {
      var i = 0
      while (i < activeJobs.size) {
        try {
          // Wait at most one second
          Await.ready(activeJobs(i), Duration.fromNanos(1E9))
          activeJobs.remove(i)
        } catch {
          case _: TimeoutException | _: InterruptedException => i += 1
        }
      }
    }

    val processor: Future[Unit] = Future {
      try {
        val dataset1Name = pair.getAs[String](0)
        val dataset2Name = pair.getAs[String](1)
        if (!existingResults.contains((dataset1Name, dataset2Name))) {
          val dataset1 = sc.readWKTFile(s"lakes_parks/$dataset1Name", "geometry")
          val cardinality1 = dataset1.count()
          val dataset2 = sc.readWKTFile(s"lakes_parks/$dataset2Name", "geometry")
          val cardinality2 = dataset2.count()
          // Run the join and calculate number of MBR tests
          val dataset1Partitioned = dataset1.spatialPartition(classOf[RSGrovePartitioner], 50, "disjoint" -> true)
          val dataset2Partitioned = dataset2.spatialPartition(classOf[RSGrovePartitioner], 50, "disjoint" -> true)
          val numMBRTests = sc.longAccumulator("mbrTests")
          val resultSize = SpatialJoin.spatialJoinDJ(dataset1Partitioned, dataset2Partitioned, ESJPredicate.MBRIntersects, numMBRTests).filter(
            pair => try {pair._1.getGeometry.intersects(pair._2.getGeometry)} catch {case e: org.locationtech.jts.geom.TopologyException => false}).count()
          outputResults.synchronized {
            outputResults.println(Array(dataset1Name, dataset2Name, cardinality1 * scale, cardinality2 * scale, resultSize * scale * scale,
              resultSize.toDouble / cardinality1 / cardinality2, numMBRTests.value * scale * scale).mkString(","))
          }
        }
      } catch {
        case e: Throwable => e.printStackTrace()
      }
    }
    activeJobs.append(processor)
  }
  // Finish any remaining jobs
  while (activeJobs.nonEmpty) {
    var i = 0
    while (i < activeJobs.size) {
      try {
        // Wait at most one second
        Await.ready(activeJobs(i), Duration.fromNanos(1E9))
        activeJobs.remove(i)
      } catch {
        case _: TimeoutException | _: InterruptedException => i += 1
      }
    }
  }
} finally {
  outputResults.close()
}
:quit
