import org.apache.spark.SparkContext


val sc: SparkContext = null
import edu.ucr.cs.bdlab.beast._

sc.readCSVPoint("inputfile.csv", "longitude", "latitude")
  .reproject(org.geotools.referencing.CRS.decode("EPSG:3857"), org.geotools.referencing.CRS.decode("EPSG:4326"))
  .saveAsCSVPoints("outputfile.csv")

import edu.ucr.cs.bdlab.beast.operations.GriddedSummary
import org.apache.hadoop.fs.Path
import org.apache.hadoop.fs.PathFilter

val summariesPath = new Path("sj_summaries")
val paths = Array("sj/gap_datasets", "sj/large_datasets", "sj/medium_datasets", "sj/real_datasets", "sj/small_datasets").map(new Path(_))
val conf = sc.hadoopConfiguration
for (path <- paths) {
  val filesystem = path.getFileSystem(conf)
  val datasets = filesystem.listStatus(path)
  for (dataset <- datasets) {
    val summaryPath = new Path(summariesPath, new Path(path.getName, dataset.getPath.getName+"_summary"))
    if (!filesystem.exists(summaryPath)) {
      println(s"Summarizing ${dataset.getPath}")
      GriddedSummary.run(Seq("separator" -> ",", "iformat" -> "envelope", "numcells" -> "128,128"),
        inputs = Array(dataset.getPath.toString),
        outputs = Array(summaryPath.toString),
        sc)
      // Move the file out of the directory
      val summaryFile = filesystem.listStatus(summaryPath, new PathFilter {
        override def accept(path: Path) = path.getName.startsWith("part")
      })
      filesystem.rename(summaryFile.head.getPath, new Path(summaryPath.toString+".csv"))
      filesystem.delete(summaryPath, true)
    } else {
      println(s"Skipping $dataset")
    }
  }
}

import scala.util.Random
import edu.ucr.cs.bdlab.beast.io.{CSVFeatureWriter, SpatialWriter}
import edu.ucr.cs.bdlab.beast.generator.{BitDistribution, DiagonalDistribution, DistributionType, GaussianDistribution, ParcelDistribution, SierpinskiDistribution, UniformDistribution}
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
  dataset.writeSpatialFile(f"dataset-${i}%03d.csv", "envelope",
    Seq(SpatialWriter.CompatibilityMode -> true, CSVFeatureWriter.FieldSeparator -> ','))
}
