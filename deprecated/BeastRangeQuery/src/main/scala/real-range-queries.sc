import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession

val spark: SparkSession = SparkSession.builder().master("local").getOrCreate()
val sc: SparkContext = spark.sparkContext
import edu.ucr.cs.bdlab.beast._

// Start copying from here
import edu.ucr.cs.bdlab.beast.geolite.{EnvelopeND, EnvelopeNDLite, GeometryReader}
import edu.ucr.cs.bdlab.beast.synopses.Summary
import java.util.concurrent.TimeoutException
import scala.concurrent.Await
import scala.concurrent.duration.Duration
import org.apache.spark.sql.Row
import scala.concurrent.Future
import java.io.{File, PrintStream}
import org.apache.spark.storage.StorageLevel
import java.io.FileOutputStream
import scala.concurrent.ExecutionContext.Implicits.global

val outputPath = "range-query-output.csv"
val conf = sc.hadoopConfiguration
// A set of all queries that ran previously
var previouslyRunQueries = Set[(String, Int)]()
val file = new File(outputPath)
val writer = new PrintStream(new FileOutputStream(file))
case class QueryResult(dataset: String, numQuery: Int, queryArea: Double, areaInt: Double, cardinality: Long, executionTime: Long, datasetLoading: Long, totalTime: Long, mbrTests: Long)
writer.println("dataset,numQuery,queryArea,areaInt,cardinality,executionTime(sec),datasetLoading(sec),totalTime(sec),mbrTests")
val queries = new collection.mutable.ArrayBuffer[Row]()
spark.read.option("delimiter", ",").option("header", true).option("inferschema", true).csv("rq_result_35925_balanced.csv").createOrReplaceTempView("queries")
queries ++= spark.sql("SELECT *, ROW_NUMBER() OVER (PARTITION BY dataset ORDER BY minx) as numQuery FROM queries").collect()
val parallelism = 32
val runningQueries = new collection.mutable.ArrayBuffer[Future[QueryResult]]()
val cachedDatasets = new collection.mutable.HashMap[String, SpatialRDD]()
val datasetLoadingTime = new collection.mutable.HashMap[String, Long]()
val cachedDatasetsCount = new collection.mutable.HashMap[String, Int]()
val datasetMBR = new collection.mutable.HashMap[String, Summary]()
try {
  while (queries.nonEmpty || runningQueries.nonEmpty) {
    // Try to finish some of the queries
    var i = 0
    while (i < runningQueries.size) {
      try {
        println(s"Checking running query #$i")
        val result: QueryResult = Await.result(runningQueries(i), Duration.fromNanos(1E9))
        println("Got result")
        writer.println(Array(
          result.dataset,
          result.numQuery,
          result.queryArea,
          result.areaInt,
          result.cardinality,
          result.executionTime * 1E-9,
          result.datasetLoading * 1E-9,
          result.totalTime * 1E-9,
          result.mbrTests
        ).mkString(","))
        runningQueries.remove(i)
        cachedDatasetsCount.put(result.dataset, cachedDatasetsCount(result.dataset) - 1)
        if (cachedDatasetsCount(result.dataset) == 0) {
          val dataset = cachedDatasets.remove(result.dataset).get
          dataset.unpersist(false)
        }
      } catch {
        case _: TimeoutException | _: InterruptedException => i += 1
      }
    }
    val geometryFactory = GeometryReader.DefaultGeometryFactory
    while (queries.nonEmpty && runningQueries.size < parallelism) {
      val queryToRun: Row = queries.remove(queries.size - 1)
      val datasetName = queryToRun.getAs[String]("dataset")
      val dataset: SpatialRDD = cachedDatasets.getOrElseUpdate(datasetName, {
        val t1 = System.nanoTime()
        val dataset = sc.readWKTFile(datasetName, 0, '\t', true)
        dataset.persist(StorageLevel.MEMORY_ONLY)
        dataset.count()
        val t2 = System.nanoTime()
        datasetMBR.put(datasetName, dataset.summary)
        datasetLoadingTime.put(datasetName, t2 - t1)
        dataset
      })
      cachedDatasetsCount.put(datasetName, cachedDatasetsCount.getOrElse(datasetName, 0) + 1)
      runningQueries.append(Future {
        try {
          val t1 = System.nanoTime()
          val x1 = queryToRun.getAs[Double]("rq_minx")
          val y1 = queryToRun.getAs[Double]("rq_miny")
          val x2 = queryToRun.getAs[Double]("rq_maxx")
          val y2 = queryToRun.getAs[Double]("rq_maxy")
          val queryMBR = new EnvelopeNDLite(2, x1, y1, x2, y2)
          val mbrCount = sc.longAccumulator("num-mbr")
          val cardinality = dataset.rangeQuery(new EnvelopeND(geometryFactory, queryMBR), mbrCount).count()
          val t2 = System.nanoTime()
          val queryTime = t2 - t1
          QueryResult(datasetName,
            queryToRun.getAs[Int]("numQuery"),
            queryMBR.getArea,
            queryMBR.intersectionEnvelope(datasetMBR(datasetName)).getArea,
            cardinality,
            queryTime,
            datasetLoadingTime(datasetName),
            queryTime + datasetLoadingTime(datasetName),
            mbrCount.value
          )
        } catch {
          case e: Exception => e.printStackTrace(); throw e
        }
      })
    }
  }
} finally {
  writer.close()
}

