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
import scala.concurrent.{Await, Future}
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration._
import java.io.{BufferedReader, FileReader}
import edu.ucr.cs.bdlab.beast.common.BeastOptions
import edu.ucr.cs.bdlab.beast.generator.{BitDistribution, DiagonalDistribution, DistributionType, GaussianDistribution, ParcelDistribution, SierpinskiDistribution, UniformDistribution}
import edu.ucr.cs.bdlab.beast.cg.SpatialJoinAlgorithms
import edu.ucr.cs.bdlab.beast.indexing.IndexHelper
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import com.fasterxml.jackson.module.scala.experimental.ScalaObjectMapper

val scale = 1

val datasetsPath: String = "../tvu032/datasets/sjml/"


def jsonToMap(jsonString: String): Map[String, Any] = {
  val mapper = new ObjectMapper() with ScalaObjectMapper
  mapper.registerModule(DefaultScalaModule)
  mapper.readValue[Map[String, Any]](jsonString)
}

def generateDataset(descriptor: Map[String, Any]): SpatialRDD = {
  val opts: BeastOptions = descriptor
  if (descriptor.contains("path")) {
    // Load from disk
    sc.spatialFile(datasetsPath+opts.getString("path"), opts.getString("format"), opts)
  } else {
    val distributions: Map[String, DistributionType] = Array(UniformDistribution, DiagonalDistribution,
      GaussianDistribution, BitDistribution, SierpinskiDistribution, ParcelDistribution)
      .map(x => (x.toString, x))
      .toMap
    sc.generateSpatialData.distribution(distributions(opts.getString("distribution")))
      .config(opts).generate(opts.getString("cardinality").toLong / scale)
  }
}

var pairs = spark.read.option("delimiter", ",").option("header", true).csv("sj-piechart-pairs.csv").select("dataset1", "dataset2").collect
val datasetDescriptors: Map[String, Map[String, Any]] = sc.textFile("sj-piechart-descriptors.json").collect.map(d => {
  val map = jsonToMap(d)
  val name = map("name").asInstanceOf[String]
  (name, map)
}).toMap
val resultFileName: String = "sj-piechart-results.csv"
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
  ps.println("dataset1,datasets2,cardinality1,cardinality2,sjresultsize,selectivity,djTime,djMBRTests,pbsmTime,pbsmMBRTests,repJTime,repJMBRTests,sj1dTime,sj1dMBRTests,bnljTime,bnljMBRTests")
  ps
}
val maxParallelism = 1
try {
  val activeJobs = new ArrayBuffer[Future[Unit]]()
  pairs = pairs.map(x => (Math.random(), x)).sortWith((x1, x2) => x1._1 < x2._1).map(x => x._2)
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
          sc.setJobGroup("persist", s"Persisting datasets $dataset1Name &#8904; $dataset2Name")
          val dataset1: SpatialRDD = generateDataset(datasetDescriptors(dataset1Name))
          dataset1.setName(dataset1Name)
          val dataset2: SpatialRDD = generateDataset(datasetDescriptors(dataset2Name))
          dataset2.setName(dataset2Name)
          // Run the join and calculate number of MBR tests
          val dataset1Partitioned = IndexHelper.partitionFeatures2(dataset1, classOf[RSGrovePartitioner], _.getStorageSize, Seq("disjoint" -> true, IndexHelper.PartitionCriterionThreshold -> "Size(16m)"))
          dataset1Partitioned.setName(s"${dataset1Name}_partitioned")
          val dataset2Partitioned = IndexHelper.partitionFeatures2(dataset2, classOf[RSGrovePartitioner], _.getStorageSize, Seq("disjoint" -> true, IndexHelper.PartitionCriterionThreshold -> "Size(16m)"))
          dataset2Partitioned.setName(s"${dataset2Name}_partitioned")
          // Persist all datasets to disregard their creation time
          val pers1f = dataset1.persist().countAsync()
          val pers2f = dataset2.persist().countAsync()
          val pers3f = dataset1Partitioned.persist().countAsync()
          val pers4f = dataset2Partitioned.persist().countAsync()
          val cardinality1: Long = Await.result(pers1f, Duration.fromNanos(1000E9))
          val cardinality2: Long = Await.result(pers2f, Duration.fromNanos(1000E9))
          Await.result(pers3f, Duration.fromNanos(1000E9))
          Await.result(pers4f, Duration.fromNanos(1000E9))
          // Try all algorithms in this order
          val algorithms = Seq(SpatialJoinAlgorithms.ESJDistributedAlgorithm.DJ,
            SpatialJoinAlgorithms.ESJDistributedAlgorithm.PBSM,
            SpatialJoinAlgorithms.ESJDistributedAlgorithm.REPJ,
            SpatialJoinAlgorithms.ESJDistributedAlgorithm.SJ1D,
            SpatialJoinAlgorithms.ESJDistributedAlgorithm.BNLJ)
          var minTime: Long = Long.MaxValue
          val processingTimes: Seq[(Long, Long, Long)] = algorithms.map(algorithm => {
            val numMBRTests = sc.longAccumulator("mbrTests")
            val inputs = algorithm match {
              case SpatialJoinAlgorithms.ESJDistributedAlgorithm.DJ => (dataset1Partitioned, dataset2Partitioned)
              case SpatialJoinAlgorithms.ESJDistributedAlgorithm.PBSM => (dataset1, dataset2)
              case SpatialJoinAlgorithms.ESJDistributedAlgorithm.REPJ => (dataset1Partitioned, dataset2)
              case SpatialJoinAlgorithms.ESJDistributedAlgorithm.SJ1D => (dataset1, dataset2)
              case SpatialJoinAlgorithms.ESJDistributedAlgorithm.BNLJ => (dataset1, dataset2)
            }
            val t1 = System.nanoTime()
            var resultSize: Long = -1
            sc.setJobGroup("process", s"Processing $dataset1Name &#8904; $dataset2Name with $algorithm")
            val resultSizeF: org.apache.spark.FutureAction[Long] = edu.ucr.cs.bdlab.beast.operations.SpatialJoin.spatialJoin(inputs._1, inputs._2,
              ESJPredicate.MBRIntersects, algorithm, numMBRTests).countAsync()
            while ((System.nanoTime() - t1) / 2 < minTime && resultSize == -1) {
              try {
                resultSize = Await.result(resultSizeF, Duration.fromNanos(1E9))
              } catch {
                case _: TimeoutException | _: InterruptedException =>
              }
            }
            val t2 = System.nanoTime()
            if (resultSize >= 0) {
              minTime = minTime min (t2 - t1)
              (resultSize, numMBRTests.value.toLong, t2 - t1)
            } else {
              resultSizeF.cancel()
              (resultSize, -1L, minTime * 10)
            }
          })
          // "dataset1,datasets2,cardinality1,cardinality2,sjresultsize,selectivity,djTime,djMBRTests,pbsmTime,pbsmMBRTests,repJTime,repJMBRTests,sj1dTime,sj1dMBRTests,bnljTime,bnljMBRTests"
          outputResults.synchronized {
            outputResults.println(Array(dataset1Name, dataset2Name, cardinality1 * scale, cardinality2 * scale,
              processingTimes(0)._1 * scale * scale, processingTimes(0)._1.toDouble / cardinality1 / cardinality2,
              processingTimes(0)._3 * 1E-9, processingTimes(0)._2 * scale * scale,
              processingTimes(1)._3 * 1E-9, processingTimes(1)._2 * scale * scale,
              processingTimes(2)._3 * 1E-9, processingTimes(2)._2 * scale * scale,
              processingTimes(3)._3 * 1E-9, processingTimes(3)._2 * scale * scale,
              processingTimes(4)._3 * 1E-9, processingTimes(4)._2 * scale * scale,
            ).mkString(","))
          }
          dataset1.unpersist()
          dataset2.unpersist()
          dataset1Partitioned.unpersist()
          dataset2Partitioned.unpersist()
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
