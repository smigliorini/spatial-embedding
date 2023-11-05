import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

val conf = new SparkConf
conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
conf.setAppName("SpatialGenerator")

// Set Spark master to local if not already set
if (!conf.contains("spark.master"))
  conf.setMaster("local[*]")

val spark: SparkSession = SparkSession.builder().config(conf).getOrCreate()
val sc: SparkContext = spark.sparkContext

import edu.ucr.cs.bdlab.beast._

// Start copying from here

import edu.ucr.cs.bdlab.beast.synopses.BoxCounting
import java.io.{File, PrintStream}
import edu.ucr.cs.bdlab.beast.operations.GriddedSummary
import org.apache.hadoop.fs.Path
import org.apache.hadoop.fs.PathFilter
import edu.ucr.cs.bdlab.beast.geolite.EnvelopeNDLite

val summariesPath = new Path("real_summaries")
// Scale all summaries from the world MBR to the scale MBR
val worldMBR = new EnvelopeNDLite(2, -180, -90, 180, 90)
val scaleMBR = new EnvelopeNDLite(2, 0, 0, 10, 10)
//val paths = Array("sj/gap_datasets", "sj/large_datasets", "sj/medium_datasets", "sj/real_datasets", "sj/small_datasets").map(new Path(_))
val paths = Array("lakes_parks").map(new Path(_))
val conf = sc.hadoopConfiguration
val globalSummaryFile = "global-summaries.csv"
val globalSummaries = new PrintStream(new File(globalSummaryFile))
globalSummaries.println("dataset,distribution,x1,y1,x2,y2,num_features,size,num_points,avg_area,avg_side_length_0,avg_side_length_1,E0,E2")
try {
  for (path <- paths) {
    val filesystem = path.getFileSystem(conf)
    val datasets = filesystem.listStatus(path)
    for (dataset <- datasets) {
      {
        // Compute global summary
        val datasetRDD = sc.readWKTFile(dataset.getPath.toString, 0, skipHeader = true)
        val summary = datasetRDD.summary
        // Scale the global summary MBR to fit within the scale MBR
        for (d <- 0 until summary.getCoordinateDimension) {
          summary.setMinCoord(d, (summary.getMinCoord(d) - worldMBR.getMinCoord(d)) / worldMBR.getSideLength(d) * scaleMBR.getSideLength(d) + scaleMBR.getMinCoord(d))
          summary.setMaxCoord(d, (summary.getMaxCoord(d) - worldMBR.getMinCoord(d)) / worldMBR.getSideLength(d) * scaleMBR.getSideLength(d) + scaleMBR.getMinCoord(d))
        }
        val bcHistogram = BoxCounting.computeBCHistogram(datasetRDD, 128)
        val e0 = Math.abs(BoxCounting.boxCounting(bcHistogram, 0))
        val e2 = BoxCounting.boxCounting(bcHistogram, 2)
        globalSummaries.println(Array(dataset.getPath, "real", summary.getMinCoord(0), summary.getMinCoord(1),
          summary.getMaxCoord(0), summary.getMaxCoord(1), summary.numFeatures, summary.size,
          summary.numPoints, summary.averageArea, summary.averageSideLength(0), summary.averageSideLength(1),
          e0, e2).mkString(","))
      }
      val summaryPath = new Path(summariesPath, new Path(path.getName, dataset.getPath.getName + "_summary.csv"))
      if (!filesystem.exists(summaryPath)) {
        println(s"Summarizing ${dataset.getPath}")
        try {
          {
            // Compute local summaries
            val tempSummaryPath = new Path(summaryPath.getParent, summariesPath.getName + "_temp")
            GriddedSummary.run(Seq("separator" -> "\t", "iformat" -> "wkt", "numcells" -> "128,128", "skipheader" -> true),
              inputs = Array(dataset.getPath.toString),
              outputs = Array(tempSummaryPath.toString),
              sc)
            // Move the file out of the directory
            val summaryFile = filesystem.listStatus(tempSummaryPath, new PathFilter {
              override def accept(path: Path): Boolean = path.getName.startsWith("part")
            })
            filesystem.rename(summaryFile.head.getPath, summaryPath)
            filesystem.delete(tempSummaryPath, true)
          }
        } catch {
          case _: Exception => System.err.println(s"Error summarizing file '$dataset'")
        }
      } else {
        println(s"Skipping ${dataset.getPath}")
      }
    }
  }
} finally {
  globalSummaries.close()
}


///////
import scala.util.Random
import edu.ucr.cs.bdlab.beast.io.{CSVFeatureWriter, SpatialWriter}
import edu.ucr.cs.bdlab.beast.generator.{BitDistribution, DiagonalDistribution, DistributionType, GaussianDistribution,
  ParcelDistribution, SierpinskiDistribution, UniformDistribution}
import edu.ucr.cs.bdlab.beast.geolite.EnvelopeNDLite

def randomDouble(random: Random, range: Array[Double]) = {
  random.nextDouble() * (range(1) - range(0)) + range(0)
}

val boxSizes: Array[Double] = Array(1E-6, 1E3)
val distributions: Array[DistributionType] =
  Array(UniformDistribution, GaussianDistribution, BitDistribution,
    ParcelDistribution, DiagonalDistribution, SierpinskiDistribution)
val cardinalities = Array(10000, 100000000)
val percentages = Array(0.1, 0.9)
val buffers = Array(0.1, 0.3)
// For bit distribution
val probabilities = Array(0.1, 0.9)
// For parcel
val splitRanges = Array(0.1, 0.5)
val dither = Array(0.1, 0.5)
val numDatasets = 1
val globalMBR = new EnvelopeNDLite(2, -10, -10, 10, 10)
val random = new Random(0)
for (i <- 1 to numDatasets) {
  val cardinality = random.nextInt(cardinalities(1) - cardinalities(0)) + cardinalities(0)
  val distribution = distributions(random.nextInt(distributions.length))
  val mbrWidth = randomDouble(random, Array(0, globalMBR.getSideLength(0)))
  val mbrHeight = randomDouble(random, Array(0, globalMBR.getSideLength(1)))
  val x1 = randomDouble(random, Array(globalMBR.getMinCoord(0), globalMBR.getMaxCoord(0) - mbrWidth))
  val y1 = randomDouble(random, Array(globalMBR.getMinCoord(1), globalMBR.getMaxCoord(1) - mbrHeight))
  val datasetMBR = new EnvelopeNDLite(2, x1, y1, x1 + mbrWidth, y1 + mbrHeight)
  val boxSize: Array[Double] = Array(randomDouble(random, boxSizes), randomDouble(random, boxSizes))
  val generator = sc.generateSpatialData.mbr(datasetMBR)
    .config(UniformDistribution.MaxSize, s"${boxSize(0)},${boxSize(1)}")
    .config(UniformDistribution.GeometryType, "box")
    .distribution(distribution)
  distribution match {
    case BitDistribution =>
      generator.config(BitDistribution.Digits, 12)
      .config(BitDistribution.Probability, randomDouble(random, probabilities))
    case ParcelDistribution =>
      generator.config(ParcelDistribution.SplitRange, randomDouble(random, splitRanges))
      .config(ParcelDistribution.Dither, randomDouble(random, dither))
    case DiagonalDistribution =>
      generator.config(DiagonalDistribution.Buffer, randomDouble(random, buffers))
      .config(DiagonalDistribution.Percentage, randomDouble(random, percentages))
  }
  val dataset: SpatialRDD = generator.generate(cardinality)
  // 1- Write the dataset to the output as a single file
  dataset.writeSpatialFile(f"dataset-$i%03d.csv", "envelope",
    Seq(SpatialWriter.CompatibilityMode -> true, CSVFeatureWriter.FieldSeparator -> ','))
}
