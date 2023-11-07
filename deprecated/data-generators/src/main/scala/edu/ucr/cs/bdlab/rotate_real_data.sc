import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession


val spark: SparkSession = SparkSession.builder().master("local").getOrCreate()
val sc: SparkContext = spark.sparkContext
import edu.ucr.cs.bdlab.beast._

// Start copying from here into Spark shell
import java.util.concurrent.TimeoutException
import scala.collection.mutable.ArrayBuffer
import org.apache.hadoop.fs.{FileSystem, Path}
import scala.util.matching.Regex
import java.awt.geom.AffineTransform
import scala.util.Random
import edu.ucr.cs.bdlab.beast.geolite.Feature
import org.geotools.geometry.jts.JTS
import scala.concurrent.{Await, Future}
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration._

val newDataPairs = spark.read.option("delimiter", ",").option("header", true).csv("realDataset_jn_pairs_rot.csv").select("dataset1", "dataset2").collect().map(r => Array(r.getAs[String](0), r.getAs[String](1)))
val rotatedDSExp: Regex = ".*_r([0-9]+).csv".r
val maxParallelism = 32
val activeJobs = new ArrayBuffer[Future[Unit]]()

for (newDataPair <- newDataPairs) {
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
    val fs = FileSystem.get(sc.hadoopConfiguration)
    val outfiles = newDataPair.map(x => new Path("lakes_parks", x))
    if (!outfiles.forall(x => fs.exists(x))) {
      val originalFiles = newDataPair.map(x => new Path("lakes_parks", x.replaceAll("_r[0-9]+", "")))
      val originalDatasets = originalFiles.map(x => sc.readWKTFile(x.toString, "geometry"))
      val mbr = sc.union(originalDatasets).summary
      // Rotate the datasets randomly and equally to make sure the number of results remain the same
      val seed: Int =  newDataPair.head match { case rotatedDSExp(i) => i.toInt }
      val random = new Random(seed)
      // Generate and discard first random number which is almost the same for all small seeds
      // https://stackoverflow.com/questions/12282628/why-are-initial-random-numbers-similar-when-using-similar-seeds
      random.nextDouble()
      val angle = random.nextDouble() * Math.PI
      val center = (mbr.getCenter(0), mbr.getCenter(1))
      val rotTransform = AffineTransform.getRotateInstance(angle, center._1, center._2)
      // Write the new files to the output
      for (i <- 0 to 1) {
        val rotatedDataset: SpatialRDD = originalDatasets(i).map(f => {
          val g = JTS.transform(f.getGeometry, new org.geotools.referencing.operation.transform.AffineTransform2D(rotTransform))
          Feature.create(f, g)
        })
        rotatedDataset.coalesce(1).saveAsWKTFile(outfiles(i).toString+"_temp", 0, '\t', true)
        fs.rename(new Path(outfiles(i).toString+"_temp", "part-00000.csv"), outfiles(i))
        fs.delete(new Path(outfiles(i).toString+"_temp"), true)
      }
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
println("Done!")
