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

import edu.ucr.cs.bdlab.beast.common.BeastOptions
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
import org.apache.spark.sql.{DataFrame, Row, SaveMode, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import org.locationtech.jts.geom.{Envelope, GeometryFactory}

import java.io.{File, FileOutputStream, PrintStream}
import java.util.concurrent.TimeoutException
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration._
import scala.concurrent.{Await, Future}
import scala.util.Random

object GenerateFromDescriptors {
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
      .addOption(new org.apache.commons.cli.Option("h", "help", false,
        "Print this help information"))
      .addOption(new org.apache.commons.cli.Option("p", "parallelism", true,
        "Level of parallelism is how many concurrent datasets are processed together"))
      .addOption(new org.apache.commons.cli.Option("ds", "data-descriptor", true,
        "A path to a file that contains the descriptors of the data to generate"))
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
    val globalSummaryOutput: PrintStream = if (!commandline.hasOption("global-summary")) null else {
      val globalSummaryPath = new File(commandline.getOptionValue("global-summary"))
      val out = new PrintStream(new FileOutputStream(globalSummaryPath))
      out.println("dataset,distribution,x1,y1,x2,y2,num_features,size,num_points,avg_area,avg_side_length_0,avg_side_length_1,E0,E2")
      out
    }
    val descriptors = spark.read.json(commandline.getOptionValue("data-descriptor")).collect()

    try {
      val datasetsToGenerate = new collection.mutable.ArrayBuffer[String]()
      datasetsToGenerate ++= descriptors.map(x => x.getAs[String]("name"))
      val datasetsBeingGenerated = new collection.mutable.ArrayBuffer[Future[String]]()
      val parallelism = commandline.getOptionValue("parallelism", "32").toInt

      if (!outPath.toString.startsWith("/dev/null"))
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
          val datasetName = datasetsToGenerate.remove(datasetsToGenerate.size - 1)
          datasetsBeingGenerated.append(Future {
            try {
              val opts = new BeastOptions()
              val datasetDescriptor = descriptors.find(x => x.getAs[String]("name") == datasetName).get
              for (i <- 0 until datasetDescriptor.length; if !datasetDescriptor.isNullAt(i)) {
                opts.set(datasetDescriptor.schema(i).name, datasetDescriptor.getAs[String](i))
              }
              val distributions: Map[String, DistributionType] = Array(UniformDistribution, DiagonalDistribution, GaussianDistribution, BitDistribution, SierpinskiDistribution, ParcelDistribution)
                .map(x => (x.toString, x))
                .toMap
              val dataset = sc.generateSpatialData.distribution(distributions(datasetDescriptor.getAs[String]("distribution")))
                .config(opts).generate(datasetDescriptor.getAs[String]("cardinality").toLong)
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
                if (!filesystem.rename(summaryFile.head.getPath, summaryPath)) {
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
                val distribution = datasetDescriptor.getAs[String]("distribution")
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
              datasetName
            }
          catch {
            case e: Exception => e.printStackTrace(); datasetName
          }})
        }
      }
    }
    finally {
      spark.stop()
      if (globalSummaryOutput != null) {
        globalSummaryOutput.flush()
        globalSummaryOutput.close()
      }
    }

  }

}
