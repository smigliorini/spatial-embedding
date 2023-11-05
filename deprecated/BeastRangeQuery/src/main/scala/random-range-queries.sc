import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession


val sc: SparkContext = null
val spark: SparkSession = null
import edu.ucr.cs.bdlab.beast._

// Start copying from here
import edu.ucr.cs.bdlab.beast.geolite.EnvelopeNDLite
import org.locationtech.jts.geom.Geometry
import org.apache.hadoop.fs.Path
import java.io.{File, PrintStream}
import org.apache.spark.storage.StorageLevel
import java.io.FileOutputStream

val outputPath = "range-query-output.csv"
val paths = Array("lakes_parks").map(new Path(_))
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

// Scale all summaries from the world MBR to the scale MBR
val worldMBR = new EnvelopeNDLite(2, -180, -90, 180, 90)
val scaleMBR = new EnvelopeNDLite(2, 0, 0, 10, 10)

def normalize(mbr: EnvelopeNDLite, globalMBR: EnvelopeNDLite, targetMBR: EnvelopeNDLite): EnvelopeNDLite = {
  val scaledMBR = new EnvelopeNDLite(mbr.getCoordinateDimension)
  for (d <- 0 until mbr.getCoordinateDimension) {
    scaledMBR.setMinCoord(d, (mbr.getMinCoord(d) - globalMBR.getMinCoord(d)) /
      globalMBR.getSideLength(d) * targetMBR.getSideLength(0) + targetMBR.getMinCoord(d))
    scaledMBR.setMaxCoord(d, (mbr.getMaxCoord(d) - globalMBR.getMinCoord(d)) /
      globalMBR.getSideLength(d) * targetMBR.getSideLength(0) + targetMBR.getMinCoord(d))
  }
  scaledMBR
}

//val rangeQueries = new PrintStream(new FileOutputStream("range-queries.csv"))
//rangeQueries.println("datasetName,numQuery,minX,minY,maxX,maxY")
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
      for (queryIndex <- 1 to 100) {
        //val queryMBR = new EnvelopeNDLite(query.getEnvelopeInternal)
        //val scaledMBR = normalize(queryMBR, worldMBR, scaleMBR)
        //rangeQueries.println(Array(datasetFileName.getPath, queryIndex, scaledMBR.getMinCoord(0), scaledMBR.getMinCoord(1),
        //  scaledMBR.getMaxCoord(0), scaledMBR.getMaxCoord(1)).mkString(","))
        if (!previouslyRunQueries.contains((datasetFileName.getPath.toString, queryIndex))) {
          val query = queries(queryIndex - 1)
          val startQueryTime = System.nanoTime()
          val mbrTests = sc.longAccumulator("mbrTests")
          val rangeQueryResult = dataset.rangeQuery(query, mbrTests)
          val cardinality = rangeQueryResult.count()
          val queryTime = System.nanoTime() - startQueryTime
          val queryArea = query.getArea / datasetMBR.getArea
          writer.println(s"${datasetFileName.getPath};${queryIndex};$queryArea,$queryArea,$cardinality,$queryTime,$datasetLoadingTime,${datasetLoadingTime + queryTime},${mbrTests.value}")
        }
      }
      dataset.unpersist(true)
    }
  }
} finally {
  writer.close()
  //rangeQueries.close()
}

