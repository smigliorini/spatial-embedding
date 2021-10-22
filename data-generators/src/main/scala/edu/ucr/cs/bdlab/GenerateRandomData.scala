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
import edu.ucr.cs.bdlab.beast.{SpatialRDD, _}
import edu.ucr.cs.bdlab.davinci.SingleLevelPlot
import org.apache.commons.cli.{BasicParser, HelpFormatter, Options}
import org.apache.hadoop.fs.{Path, PathFilter}
import org.apache.spark.sql.{DataFrame, Row, SaveMode, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import org.locationtech.jts.geom.{Envelope, GeometryFactory}

import java.io.PrintStream
import java.util.concurrent.TimeoutException
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration._
import scala.concurrent.{Await, Future}
import scala.util.Random

object GenerateRandomData {
  val globalMBR = new EnvelopeNDLite(2, 0, 0, 10, 10)

  lazy val commandLineOptions: Options = {
    new Options()
      .addOption(new org.apache.commons.cli.Option("o", "output", true,
        "The path to write all output files"))
      .addOption(new org.apache.commons.cli.Option("d", "datasets", true,
        "The path to write the output datasets to (relative to the output directory)"))
      .addOption(new org.apache.commons.cli.Option("s", "summaries", true,
        "The path to write the summaries (relative to the output directory)"))
      .addOption(new org.apache.commons.cli.Option("gs", "global-summary", true,
        "The path to write the file with global summary of all datasets to (relative to the output directory)"))
      .addOption(new org.apache.commons.cli.Option("i", "images", true,
        "The path to write the images (relative to the output directory)"))
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
    val outPath = new Path(commandline.getOptionValue("output", "embedding-assorted-datasets"))
    val filesystem = outPath.getFileSystem(sc.hadoopConfiguration)
    val datasetsPath = new Path(outPath, commandline.getOptionValue("datasets", "datasets"))
    val summariesPath = new Path(outPath, commandline.getOptionValue("summaries", "summaries"))
    val imagePath = new Path(outPath, commandline.getOptionValue("images", "images"))
    val summaryOutput: PrintStream = if (!commandline.hasOption("global-summary")) null else {
      val globalSummaryPath = new Path(outPath, commandline.getOptionValue("global-summary"))
      val out = new PrintStream(filesystem.create(globalSummaryPath))
      out.println("dataset,x1,y1,x2,y2,num_features,size,num_points,avg_area,avg_side_length_0,avg_side_length_1")
      out
    }
    val queriesInput = commandline.getOptionValue("queries-input")
    var existingResults: Array[Row] = Array()
    val queriesOutput: PrintStream = if (queriesInput == null) null else {
      val queriesOutputFile = new Path(outPath, commandline.getOptionValue("queries-output", "range-queries-result.csv"))
      val outFS = queriesOutputFile.getFileSystem(sc.hadoopConfiguration)
      if (outFS.exists(queriesOutputFile)) {
        // Read existing results, if any
        existingResults = spark.read
          .option("delimiter", ";").option("header", true).option("inferschema", true)
          .csv(outPath.toString).collect()
      }
      val out = new PrintStream(outFS.create(queriesOutputFile))
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
      val numDatasets = 2000
      val datasetsToGenerate = new collection.mutable.ArrayBuffer[Int]()
      datasetsToGenerate ++= 1 to numDatasets
      val datasetsBeingGenerated = new collection.mutable.ArrayBuffer[Future[Int]]()
      val parallelism = commandline.getOptionValue("parallelism", "32").toInt

      outPath.getFileSystem(sc.hadoopConfiguration).mkdirs(outPath)

      while (datasetsToGenerate.nonEmpty || datasetsBeingGenerated.nonEmpty) {
        // Check if any jobs are done
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
          val i = datasetsToGenerate.remove(datasetsToGenerate.size - 1)
          datasetsBeingGenerated.append(Future {
            val dataset = generateDataset(sc, i)
            val datasetName = f"dataset-$i%04d"
            // 1- Write the dataset to the output as a single file
            val datasetFile = new Path(datasetsPath, datasetName + ".wkt.bz2")
            if (!datasetFile.toString.startsWith("/dev/null") && !filesystem.isFile(datasetFile)) {
              dataset.writeSpatialFile(datasetFile.toString, "wkt",
                Seq(SpatialWriter.CompatibilityMode -> true, CSVFeatureWriter.FieldSeparator -> '\t',
                  SpatialWriter.OverwriteOutput -> true))
            }

            // 2- Generate summary
            val summaryPath = new Path(summariesPath, datasetName + "_summary.csv")
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
              filesystem.rename(summaryFile.head.getPath, summaryPath)
              filesystem.delete(tempSummaryPath, true)
              globalSummary
            } else {
              dataset.summary
            }

            if (summaryOutput != null) summaryOutput.synchronized {
              summaryOutput.println(s"${datasetName},${globalSummary.getMinCoord(0)},${globalSummary.getMinCoord(1)}," +
                s"${globalSummary.getMaxCoord(0)},${globalSummary.getMaxCoord(1)}," +
                s"${globalSummary.numFeatures},${globalSummary.size}," +
                s"${globalSummary.numPoints},${globalSummary.averageSideLength(0)},${globalSummary.averageSideLength(1)}," +
                s"${globalSummary.averageArea}")
            }

            // 3- Draw an image of it
            val imageFile = new Path(imagePath, datasetName + ".png")
            if (!imageFile.toString.startsWith("/dev/null") && !filesystem.isFile(imageFile))
              SingleLevelPlot.plotFeatures(dataset, 1024, 1024,
                imageFile.toString, canvasMBR = globalMBR)

            // 4- Run range queries
            if (rangeQueries != null) {
              // Select all the matching queries
              val queries: Array[Row] = rangeQueries.filter(s"datasetName='${datasetName}'").collect()
              if (queries.nonEmpty) {
                // Index the dataset
                val indexedDataset = dataset.spatialPartition(classOf[RSGrovePartitioner],
                  opts = RSGrovePartitioner.ExpandToInfinity -> false).persist()
                val geometryFactory = new GeometryFactory()
                var iQuery: Int = 1
                queries.foreach(row => {
                  // Run the query only if it does not already exist in the results
                  if (!existingResults.exists(r => r.getAs[Int]("numQuery") == iQuery && r.getAs[String]("dataset") == datasetName)) {
                    val x1 = row.getAs[Double]("minX")
                    val y1 = row.getAs[Double]("minY")
                    val x2 = row.getAs[Double]("maxX")
                    val y2 = row.getAs[Double]("maxY")
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
            i
          })
        }
      }
    }
    finally {
      spark.stop()
      if (queriesOutput != null)
        queriesOutput.close()
      if (summaryOutput != null)
        summaryOutput.close()
    }

  }

  def generateDataset(sc: SparkContext, i: Int): SpatialRDD = {
    import edu.ucr.cs.bdlab.beast._
    def randomDouble(random: Random, range: Array[Double]) = {
      random.nextDouble() * (range(1) - range(0)) + range(0)
    }

    val boxSizes: Array[Double] = Array(1E-5, 0.1)
    val distributions: Array[DistributionType] =
      Array(UniformDistribution, GaussianDistribution, BitDistribution,
        ParcelDistribution, DiagonalDistribution, SierpinskiDistribution)
    val cardinalities = Array(50000, 50000000)
    val numSegments = Array(5, 50)
    val percentages = Array(0.5, 0.9)
    val buffers = Array(0.05, 0.3)
    // For bit distribution
    val probabilities = Array(0.5, 0.9)
    // For parcel
    val splitRanges = Array(0.1, 0.5)
    val dither = Array(0.1, 0.5)

    val random = new Random(i)
    val distribution: DistributionType = if (i <= 1000) UniformDistribution
    else if (i <= 1200) DiagonalDistribution
    else if (i <= 1400) GaussianDistribution
    else if (i <= 1600) ParcelDistribution
    else if (i <= 1800) BitDistribution
    else if (i <= 2000) SierpinskiDistribution
    else null
    val cardinality = random.nextInt(cardinalities(1) - cardinalities(0)) + cardinalities(0)
    val mbrWidth = randomDouble(random, Array(1.0, globalMBR.getSideLength(0)))
    val mbrHeight = randomDouble(random, Array(1.0, globalMBR.getSideLength(1)))
    val x1 = randomDouble(random, Array(globalMBR.getMinCoord(0), globalMBR.getMaxCoord(0) - mbrWidth))
    val y1 = randomDouble(random, Array(globalMBR.getMinCoord(1), globalMBR.getMaxCoord(1) - mbrHeight))
    val datasetMBR = new EnvelopeNDLite(2, x1, y1, x1 + mbrWidth, y1 + mbrHeight)
    val generator = sc.generateSpatialData.mbr(datasetMBR)
      .config(UniformDistribution.MaxSize, s"${randomDouble(random, boxSizes) / (mbrWidth max mbrHeight)}")
      .config(UniformDistribution.NumSegments, s"${random.nextInt(numSegments(1) - numSegments(0)) + numSegments(0)}")
      .config(UniformDistribution.GeometryType, "polygon")
      .config(SpatialGenerator.Seed, i)
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
      case _ => // Nothing need to be done but the case has to be added to avoid no match exception
    }
    generator.generate(cardinality)
  }

}
