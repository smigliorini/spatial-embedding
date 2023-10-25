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
import edu.ucr.cs.bdlab.beast.cg.SpatialJoinAlgorithms.ESJPredicate
import edu.ucr.cs.bdlab.beast.indexing.RSGrovePartitioner
import edu.ucr.cs.bdlab.beast.operations.SpatialJoin
import edu.ucr.cs.bdlab.beast.generator.{BitDistribution, DiagonalDistribution, DistributionType, GaussianDistribution, ParcelDistribution, SierpinskiDistribution, UniformDistribution}
import edu.ucr.cs.bdlab.beast.common.BeastOptions
import org.apache.spark.sql.Row
import scala.concurrent.{Await, Future}
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration._
import java.io.{BufferedReader, FileReader}

def generateDataset(descriptor: Row): SpatialRDD = {
  val opts = new BeastOptions()
  for (i <- 0 until descriptor.length; if !descriptor.isNullAt(i)) {
    opts.set(descriptor.schema(i).name, descriptor.getAs[String](i))
  }
  val distributions: Map[String, DistributionType] = Array(UniformDistribution, DiagonalDistribution, GaussianDistribution, BitDistribution, SierpinskiDistribution, ParcelDistribution)
    .map(x => (x.toString, x))
    .toMap
  val dataset = sc.generateSpatialData.distribution(distributions(descriptor.getAs[String]("distribution")))
    .config(opts).generate(descriptor.getAs[String]("cardinality").toLong)
  dataset
}

val descriptors: Map[String, Row] = spark.read.json("jn_balanced_rotated_descriptors.json").collect().map(x => (x.getAs[String]("name"), x)).toMap
val pairs = spark.read.option("delimiter", ",").option("header", true).csv("jn_balanced_2023-01-19-withClasses_rot.csv").select("dataset1", "dataset2").collect
val resultFileName: String = "sj_results_revision.csv"
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
  ps.println("dataset1,datasets2,cardintality1,cardinality2,djresultsize,selectivity,djmbrTests")
  ps
}
val maxParallelism = 1
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
      val dataset1Name = pair.getAs[String](0)
      val dataset2Name = pair.getAs[String](1)
      if (!existingResults.contains((dataset1Name, dataset2Name))) {
        val dataset1 = generateDataset(descriptors(dataset1Name))
        val cardinality1 = dataset1.count()
        val dataset2 = generateDataset(descriptors(dataset2Name))
        val cardinality2 = dataset2.count()
        // Run the join and calculate number of MBR tests
        val dataset1Partitioned = dataset1.spatialPartition(classOf[RSGrovePartitioner], 50, "disjoint" -> true)
        val dataset2Partitioned = dataset2.spatialPartition(classOf[RSGrovePartitioner], 50, "disjoint" -> true)
        val numMBRTests = sc.longAccumulator("mbrTests")
        val resultSize = SpatialJoin.spatialJoinDJ(dataset1Partitioned, dataset2Partitioned, ESJPredicate.Intersects, numMBRTests).count
        println(Array(dataset1Name, dataset2Name, cardinality1, cardinality2, resultSize,
          resultSize.toDouble / cardinality1 / cardinality2, numMBRTests.value).mkString(","))
        outputResults.println(Array(dataset1Name, dataset2Name, resultSize, numMBRTests.value).mkString(","))
      }
    }
    activeJobs.append(processor)
  }
  // Finish any remaining jobs
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
} finally {
  outputResults.close()
}
