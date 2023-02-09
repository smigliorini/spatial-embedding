/*
 * Copyright 2021 University of California, Riverside
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package edu.ucr.cs.bdlab

import edu.ucr.cs.bdlab.beast.generator._
import edu.ucr.cs.bdlab.beast.geolite.EnvelopeNDLite
import edu.ucr.cs.bdlab.beast.indexing.RSGrovePartitioner
import edu.ucr.cs.bdlab.beast.io.{CSVFeatureWriter, SpatialWriter}
import edu.ucr.cs.bdlab.beast.operations.GriddedSummary
import edu.ucr.cs.bdlab.beast.synopses.BoxCounting
import edu.ucr.cs.bdlab.beast.{SpatialRDD, _}
import edu.ucr.cs.bdlab.davinci.SingleLevelPlot
import org.apache.commons.cli.{BasicParser, HelpFormatter, Options}
import org.apache.hadoop.fs.{FileSystem, Path, PathFilter}
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SaveMode, SparkSession}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}
import org.locationtech.jts.geom.{Envelope, GeometryFactory}

import java.io.{File, FileOutputStream, PrintStream}
import java.util.concurrent.TimeoutException
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration._
import scala.concurrent.{Await, Future}
import scala.util.Random

/**
 * This main function generates a slightly modified version of the random data generated in [[GenerateRandomData]].
 * It overrides the box size parameters to produce bigger geometries.
 */
object GenerateModifiedRandomData {
  val globalMBR = new EnvelopeNDLite(2, 0, 0, 10, 10)

  val commandLineOptions: Options = {
    new Options()
      .addOption(new org.apache.commons.cli.Option("o", "output", true,
        "The path to write all output files"))
      .addOption(new org.apache.commons.cli.Option("d", "datasets", true,
        "The path to write the output datasets to (relative to the output directory)"))
      .addOption(new org.apache.commons.cli.Option("s", "summaries", true,
        "The path to write the summaries (relative to the output directory)"))
      .addOption(new org.apache.commons.cli.Option("gs", "global-summary", true,
        "The path to write the file with global summary of all datasets to (relative to the working dir)"))
      .addOption(new org.apache.commons.cli.Option("i", "images", true,
        "The path to write the images (relative to the output directory)"))
      .addOption(new org.apache.commons.cli.Option("ds", "data-specs", true,
        "A path to a file that contains the specs of the data to generate"))
      .addOption(new org.apache.commons.cli.Option("qi", "queries-input", true,
        "A path to a file that contains all range queries to execute"))
      .addOption(new org.apache.commons.cli.Option("qo", "queries-output", true,
        "A path to a file to write the results of range queries to (relative to the output directory)"))
      .addOption(new org.apache.commons.cli.Option("h", "help", false,
        "Print this help information"))
      .addOption(new org.apache.commons.cli.Option("p", "parallelism", true,
        "Level of parallelism is how many concurrent datasets are processed together"))
  }

  def main(args: Array[String]): Unit = {
    // Create the Spark context
    val conf = new SparkConf
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.setAppName("SpatialGenerator")

    // Set Spark master to local if not already set
    if (!conf.contains("spark.master"))
      conf.setMaster("local[*]")

    val parser = new BasicParser
    val commandline = parser.parse(commandLineOptions, args)
    if (commandline.hasOption("help")) {
      new HelpFormatter().printHelp("data-generator", commandLineOptions)
    }
    val spark: SparkSession = SparkSession.builder().config(conf).getOrCreate()
    val sc: SparkContext = spark.sparkContext
    val outPath = new Path(commandline.getOptionValue("output", "embedding-assorted-datasets-modified"))
    val filesystem = outPath.getFileSystem(sc.hadoopConfiguration)
    val datasetsPath = new Path(outPath, commandline.getOptionValue("datasets", "datasets"))
    val summariesPath = new Path(outPath, commandline.getOptionValue("summaries", "summaries"))
    val imagePath = new Path(outPath, commandline.getOptionValue("images", "images"))
    val globalSummaryOutput: PrintStream = if (!commandline.hasOption("global-summary")) null else {
      val globalSummaryPath = new File(commandline.getOptionValue("global-summary"))
      val out = new PrintStream(new FileOutputStream(globalSummaryPath))
      out.println("dataset,distribution,x1,y1,x2,y2,num_features,size,num_points,avg_area,avg_side_length_0,avg_side_length_1,E0,E2")
      out
    }
    val datasetDescriptors: PrintStream = new PrintStream(new FileOutputStream("dataset-descriptors.json"))
    val queriesInput = commandline.getOptionValue("queries-input")
    var existingResults: Array[Row] = Array()
    val queriesOutput: PrintStream = if (queriesInput == null) null else {
      val queriesOutputFile = new File(commandline.getOptionValue("queries-output", "range-queries-result.csv"))

      if (queriesOutputFile.exists()) {
        // Read existing results, if any
        existingResults = spark.read
          .option("delimiter", ";").option("header", true).option("inferschema", true)
          .csv(outPath.toString).collect()
      }
      val out = new PrintStream(new FileOutputStream(queriesOutputFile))
      out.println("dataset;numQuery;queryArea;areaInt;cardinality;executionTime;mbrTests")
      existingResults.foreach(row => out.println(row.mkString(";")))
      out.flush()
      out
    }
    val rangeQueries: DataFrame = if (queriesInput == null) null else
      spark.read
        .option("delimiter", ",")
        .option("header", true)
        .option("inferschema", true)
        .csv(queriesInput)

    try {
      // Read the specs of the data to generate
      // Columns of importance to use:
      // ds1, ds2: The dataset names to generate in the form of 'dataset-xxxx[_yyy]'.
      //   xxxx is the seed to use for generation
      //   _yyy addition suffix that indicates we should override the generated data.
      // avgLenX1,avgLenX2,avgLenY1,avgLenY2: The average length along the x and y dimensions for both datasets
      //   This information is used to override the generated data.
      val dataSpecsFilename = commandline.getOptionValue("data-specs")
      if (dataSpecsFilename == null)
        throw new RuntimeException("data-specs file name must be provided")
      val dataSpecs = spark.read
        .option("delimiter", ",")
        .option("header", true)
        .option("inferschema", true)
        .csv(dataSpecsFilename)
        .persist(StorageLevel.MEMORY_ONLY)
      dataSpecs.createOrReplaceTempView("dataspecs")

      val datasetsToGenerate = new collection.mutable.ArrayBuffer[(String, Double)]()
      datasetsToGenerate ++= (spark sql
        """
           SELECT DISTINCT * FROM (
               (SELECT ds1 AS ds, 5*array_max(Array(avgLenX1, avgLenY1)) AS avgSize FROM dataspecs)
               UNION
               (SELECT ds2 AS ds, 5*array_max(Array(avgLenX2, avgLenY2)) AS avgSize FROM dataspecs)
           )
          """).collect.map(r => (r.getAs[String](0), r.getAs[Double](1)))
      val datasetsBeingGenerated = new collection.mutable.ArrayBuffer[Future[String]]()
      val parallelism = commandline.getOptionValue("parallelism", "32").toInt

      if (!outPath.toString.startsWith("/dev/null"))
        outPath.getFileSystem(sc.hadoopConfiguration).mkdirs(outPath)

      while (datasetsToGenerate.nonEmpty || datasetsBeingGenerated.nonEmpty) {
        // Clean up any done jobs
        var i = 0
        while (i < datasetsBeingGenerated.size) {
          val datasetGenerationProcess = datasetsBeingGenerated(i)
          try {
            // Wait at most one second
            Await.ready(datasetGenerationProcess, Duration.fromNanos(1E9))
            datasetsBeingGenerated.remove(i)
          } catch {
            case _: TimeoutException | _: InterruptedException => i += 1
          }
        }
        // Launch new jobs
        while (datasetsToGenerate.nonEmpty && datasetsBeingGenerated.size < parallelism) {
          val ds: (String, Double) = datasetsToGenerate.remove(datasetsToGenerate.size - 1)
          datasetsBeingGenerated.append(Future {
            try {
              val datasetName = ds._1
              val i: Int = datasetName.substring(8, 12).toInt
              // Generate the random data with or without override depending on whether the name is extended or not
            val dataset = if (datasetName.length > 12)
              GenerateRandomData.generateDataset(sc, i, ds._2)
            else
              GenerateRandomData.generateDataset(sc, i)
            if (datasetDescriptors != null) {
              datasetDescriptors.synchronized {
                val info: Row = dataset.asInstanceOf[RandomSpatialRDD].descriptor
                val values: Array[Object] = dataset +: Row.unapplySeq(info).toArray
                val schema: Seq[StructField] = StructField("name", StringType) +: info.schema
                val finalRow = new GenericRowWithSchema(values.toArray, StructType(schema))
                datasetDescriptors.println(finalRow.json)
              }
            }

            // 1- Write the dataset to the output as a single file
            val datasetFile = new Path(datasetsPath, datasetName + ".wkt.bz2")
            if (!datasetFile.toString.startsWith("/dev/null") && !filesystem.isFile(datasetFile)) {
              dataset.writeSpatialFile(datasetFile.toString, "wkt",
                Seq(SpatialWriter.CompatibilityMode -> true, CSVFeatureWriter.FieldSeparator -> '\t',
                  SpatialWriter.OverwriteOutput -> true))
            }

            // 2- Generate summary
            val summaryPath = new Path(summariesPath, datasetName + "_summary.csv")
            val tempfs: FileSystem = summaryPath.getFileSystem(sc.hadoopConfiguration)
            if (!tempfs.exists(summaryPath)) {
              tempfs.mkdirs(summariesPath)
            }

            val globalSummary = if (!summaryPath.toString.startsWith("/dev/null") && !filesystem.isFile(summaryPath)) {
              val tempSummaryPath = new Path(outPath, datasetName + "_summary_temp")
              val (globalSummary, localSummaries) = GriddedSummary.computeForFeatures(dataset, 128, 128)
              // Write the local summaries to the given input file
              val localSummariesDF = GriddedSummary.createSummaryDataframe(globalSummary, localSummaries)
              localSummariesDF.write
                .option("delimiter", ",")
                .option("header", true)
                .mode(SaveMode.Overwrite)
                .csv(tempSummaryPath.toString)
              // Move the file out of the directory
              val summaryFile = filesystem.listStatus(tempSummaryPath, new PathFilter {
                override def accept(path: Path): Boolean = path.getName.startsWith("part")
              })
              if (!filesystem.rename(summaryFile.head.getPath, summaryPath)){
                throw new RuntimeException(s"Could not move summary file ${summaryFile.head.getPath} to ${summaryPath}")
              }
              filesystem.delete(tempSummaryPath, true)
              globalSummary
            } else {
              dataset.summary
            }

            if (globalSummaryOutput != null) {
              // Compute box counting summaries (E0, E2)
              val bcHistogram = BoxCounting.computeBCHistogram(dataset, 128, globalSummary)
              val e0 = BoxCounting.boxCounting(bcHistogram, 0)
              val e2 = BoxCounting.boxCounting(bcHistogram, 2)
              val distribution: DistributionType = if (i <= 1000) UniformDistribution
              else if (i <= 1200) DiagonalDistribution
              else if (i <= 1400) GaussianDistribution
              else if (i <= 1600) ParcelDistribution
              else if (i <= 1800) BitDistribution
              else if (i <= 2000) SierpinskiDistribution
              else null
              globalSummaryOutput.synchronized {
                  val s = s"${datasetName},${distribution},${globalSummary.getMinCoord(0)},${globalSummary.getMinCoord(1)}," +
                    s"${globalSummary.getMaxCoord(0)},${globalSummary.getMaxCoord(1)}," +
                    s"${globalSummary.numFeatures},${globalSummary.size}," +
                    s"${globalSummary.numPoints},${globalSummary.averageSideLength(0)},${globalSummary.averageSideLength(1)}," +
                    s"${globalSummary.averageArea},${e0},${e2}"
                  globalSummaryOutput.println(s)
                  println(s"Printed '${s}'")
              }
            }

            // 3- Draw an image of it
            val imageFile = new Path(imagePath, datasetName + ".png")
            if (!imageFile.toString.startsWith("/dev/null") && !filesystem.isFile(imageFile))
              SingleLevelPlot.plotFeatures(dataset, 1024, 1024,
                imageFile.toString, canvasMBR = globalMBR)

            // 4- Run range queries
            if (rangeQueries != null) {
              // Select all the matching queries
              val queries: Array[Row] = rangeQueries.filter(s"dataset='${datasetName}'").collect()
              if (queries.nonEmpty) {
                // Index the dataset
                val indexedDataset = dataset.spatialPartition(classOf[RSGrovePartitioner],
                  opts = RSGrovePartitioner.ExpandToInfinity -> false).persist()
                val geometryFactory = new GeometryFactory()
                var iQuery: Int = 1
                queries.foreach(row => {
                  // Run the query only if it does not already exist in the results
                  if (!existingResults.exists(r => r.getAs[Int]("numQuery") == iQuery && r.getAs[String]("dataset") == datasetName)) {
                    val x1 = row.getAs[Double]("rq_minx")
                    val y1 = row.getAs[Double]("rq_miny")
                    val x2 = row.getAs[Double]("rq_maxx")
                    val y2 = row.getAs[Double]("rq_maxy")
                    val numMBRTests = sc.longAccumulator
                    val query = geometryFactory.toGeometry(new Envelope(x1, x2, y1, y2))
                    val resultSize = indexedDataset.rangeQuery(query, numMBRTests).count()
                    queriesOutput.synchronized {
                      queriesOutput.println(Array(datasetName, iQuery, query.getArea,
                        query.getEnvelopeInternal.intersection(globalSummary.toJTSEnvelope).getArea,
                        resultSize, "--", numMBRTests.value).mkString(";"))
                      queriesOutput.flush()
                    }
                  }
                  iQuery += 1
                })
                indexedDataset.unpersist()
              }
            }
            datasetName
          }
          catch {
            case e: Exception => e.printStackTrace(); ds._1
          }})
        }
      }
    }
    finally {
      spark.stop()
      if (queriesOutput != null)
        queriesOutput.close()
      if (datasetDescriptors != null)
        datasetDescriptors.close()
      if (globalSummaryOutput != null) {
        globalSummaryOutput.flush()
        globalSummaryOutput.close()
      }
    }

  }

}
