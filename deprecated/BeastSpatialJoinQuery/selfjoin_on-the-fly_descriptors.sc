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
import edu.ucr.cs.bdlab.beast.generator.{BitDistribution, DiagonalDistribution, DistributionType, GaussianDistribution, ParcelDistribution, SierpinskiDistribution, UniformDistribution}
import edu.ucr.cs.bdlab.beast.common.BeastOptions
import org.apache.spark.sql.Row
import scala.concurrent.{Await, Future}
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration._
import java.io.{BufferedReader, FileReader}
import edu.ucr.cs.bdlab.beast.cg.SpatialJoinAlgorithms.ESJPredicate

val scale = 10

def generateDataset(descriptor: Row): SpatialRDD = {
  val opts = new BeastOptions()
  for (i <- 0 until descriptor.length; if !descriptor.isNullAt(i)) {
    opts.set(descriptor.schema(i).name, descriptor.getAs[String](i))
  }
  val distributions: Map[String, DistributionType] = Array(UniformDistribution, DiagonalDistribution, GaussianDistribution, BitDistribution, SierpinskiDistribution, ParcelDistribution)
    .map(x => (x.toString, x))
    .toMap
  val dataset = sc.generateSpatialData.distribution(distributions(descriptor.getAs[String]("distribution")))
    .config(opts).generate(descriptor.getAs[String]("cardinality").toLong / scale)
  dataset
}

val descriptors: Map[String, Row] = spark.read.json("jn_balanced_rotated_descriptors.json").collect().map(x => (x.getAs[String]("name"), x)).toMap
val resultFileName: String = "selfjoin_results_revision.csv"
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
  for (descriptor <- descriptors) {
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
      val datasetName = descriptor._1
      if (!existingResults.contains(datasetName)) {
        val dataset = generateDataset(descriptor._2)
          .spatialPartition(classOf[edu.ucr.cs.bdlab.beast.indexing.RSGrovePartitioner], 50, "disjoint"->true)
        val cardinality = dataset.count()
        // Run the join and calculate number of MBR tests
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
