import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
val conf = new SparkConf()
val spark: SparkSession = null
val sc: SparkContext = null
import edu.ucr.cs.bdlab.beast._

// Start copying from here
//sc.readCSVPoint("inputfile.csv", "longitude", "latitude")
//  .reproject(org.geotools.referencing.CRS.decode("EPSG:3857"), org.geotools.referencing.CRS.decode("EPSG:4326"))
//  .saveAsCSVPoints("outputfile.csv")
import java.io.{File, FileOutputStream, PrintStream}
import java.util.concurrent.TimeoutException
import scala.collection.mutable.ArrayBuffer
import scala.concurrent.{Await, Future}
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration._
import java.io.{BufferedReader, FileReader}
import edu.ucr.cs.bdlab.beast.cg.SpatialJoinAlgorithms.ESJPredicate
import org.apache.hadoop.fs.Path

val scale = 1
val datasetPath = new Path("lakes_parks")
val fs = datasetPath.getFileSystem(sc.hadoopConfiguration)
val files = fs.listStatus(datasetPath).map(_.getPath)
val resultFileName: String = "selfjoin_real_revision.csv"
var existingResults: Array[String] = Array()
val outputResults: PrintStream = if (new File(resultFileName).exists()) {
  val file = new BufferedReader(new FileReader(resultFileName))
  var line: String = null
  do {
    line = file.readLine()
    if (line != null) {
      val split = line.trim.split(",")
      existingResults +:= split(0)
    }
  } while (line != null)
  file.close()
  new PrintStream(new FileOutputStream(resultFileName, true))
} else {
  val ps = new PrintStream(new FileOutputStream(resultFileName))
  ps.println("dataset,cardinality,selfjoin_resultsize,selectivity,selfjoinMBRTests,SelectivityAnalytical,DJCostAnalytical")
  ps
}
val maxParallelism = 32
try {
  val activeJobs = new ArrayBuffer[Future[Unit]]()
  for (file <- files) {
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
      val datasetName = file.getName
      if (!existingResults.contains(datasetName)) {
        var dataset = sc.readWKTFile(file.toString, "geometry")
        if (scale > 1)
          dataset = dataset.sample(false, 1.0f / scale)
        dataset = dataset.spatialPartition(classOf[edu.ucr.cs.bdlab.beast.indexing.RSGrovePartitioner], 50, "disjoint"->true)
        val cardinality = dataset.count()
        // Run the self join and calculate number of MBR tests
        val numMBRTests = sc.longAccumulator("mbrTests")
        val resultSize = edu.ucr.cs.bdlab.beast.operations.SpatialJoin.selfJoinDJ(dataset, ESJPredicate.Intersects, numMBRTests).count()
        // Compute the cost using the analytical query found in https://doi.org/10.1007/s10707-020-00414-x
        val summary: edu.ucr.cs.bdlab.beast.synopses.Summary = dataset.summary
        val selectivityEstimation: Double = edu.ucr.cs.bdlab.beast.synopses.Summary.spatialJoinSelectivityEstimation(summary, summary)
        val partitionStats: Array[edu.ucr.cs.bdlab.beast.synopses.Summary] = dataset.spatialPartition(classOf[edu.ucr.cs.bdlab.beast.indexing.RSGrovePartitioner])
          .mapPartitions(fs => Some(fs.summary).iterator).collect()
        var estimatedCostAnalytical: Double = 0.0
        for (p <- partitionStats) {
          // Compute selectivity of the result and use as an approximation for number of MBR tests
          val selectivity = edu.ucr.cs.bdlab.beast.synopses.Summary.spatialJoinSelectivityEstimation(p, p)
          // Calculate the cost of planesweep between the two partitions p1 and p2 using Estimate 3 (Section 5) in the paper
          val numMBRTests = p.numFeatures * p.numFeatures * selectivity
          estimatedCostAnalytical += numMBRTests
        }

        outputResults.synchronized {
          outputResults.println(Array(datasetName, cardinality * scale, resultSize * scale * scale,
            resultSize.toDouble / cardinality / cardinality, numMBRTests.value * scale * scale, selectivityEstimation, estimatedCostAnalytical * scale * scale).mkString(","))
        }
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