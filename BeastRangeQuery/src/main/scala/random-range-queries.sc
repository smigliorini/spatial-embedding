import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession


val sc: SparkContext = null
val spark: SparkSession = null
import edu.ucr.cs.bdlab.beast._

// Start copying from here
import org.locationtech.jts.geom.Geometry
import org.apache.hadoop.fs.Path
import java.io.{File, PrintStream}
import org.apache.spark.storage.StorageLevel
import java.io.FileOutputStream

val outputPath = "range-query-output.csv"
val paths = Array("parks_partitioned", "lakes_partitioned").map(new Path(_))
val conf = sc.hadoopConfiguration
// A set of all queries that ran previously
var previouslyRunQueries = Set[(String, Int)]()
val file = new File(outputPath)
val writer: PrintStream = if (file.exists()) {
  // Load all existing queries and save them to avoid running them again
  previouslyRunQueries = spark.read
    .option("delimiter", ",")
    .option("header", true)
    .csv("file://"+file.getAbsolutePath)
    .select("dataset;numQuery;queryArea")
    .collect()
    .map(_.getAs[String](0))
    .map(x => {
      val parts = x.split(";")
      (parts(0), parts(1).toInt)
    })
    .toSet
  // Append to existing file
  val ps = new PrintStream(new FileOutputStream(file, true))
  ps
} else {
  // Create a new file
  val ps = new PrintStream(new FileOutputStream(file))
  ps.println("dataset;numQuery;queryArea,areaInt,cardinality,executionTime(ms),datasetLoading(ms),totalTime(ms),mbrTests")
  ps
}

try {
  for (path <- paths) {
    val filesystem = path.getFileSystem(conf)
    val datasets = filesystem.listStatus(path)

    for (datasetFileName <- datasets) {
      val startTime = System.nanoTime()
      lazy val (dataset, datasetMBR, datasetLoadingTime, queries) = {
        val dataset = sc.readWKTFile(datasetFileName.getPath.toString, 0, skipHeader = true).persist(StorageLevel.MEMORY_AND_DISK)
        val datasetMBR = dataset.summary
        val datasetLoadingTime = System.nanoTime() - startTime
        // Generate 100 random queries
        val queries: Array[Geometry] = sc.generateSpatialData
          .config("seed", 0)
          .mbr(datasetMBR)
          .parcel(100, 0.0, 0.3)
          .map(f => f.getGeometry.getEnvelope)
          .collect()
        (dataset, datasetMBR, datasetLoadingTime, queries)
      }

      var queryIndex = 0
      for (query <- queries) {
        queryIndex += 1
        if (!previouslyRunQueries.contains((datasetFileName.getPath.toString, queryIndex))) {
          val startQueryTime = System.nanoTime()
          val mbrTests = sc.longAccumulator("mbrTests")
          val rangeQueryResult = dataset.rangeQuery(query, mbrTests)
          val cardinality = rangeQueryResult.count()
          val queryTime = System.nanoTime() - startQueryTime
          val queryArea = query.getArea / datasetMBR.getArea
          writer.println(s"${datasetFileName.getPath};${queryIndex};$queryArea,$queryArea,$cardinality,$queryTime,$datasetLoadingTime,${datasetLoadingTime + queryTime}")
        }
      }
      dataset.unpersist(true)
    }
  }
} finally {
  writer.close()
}

