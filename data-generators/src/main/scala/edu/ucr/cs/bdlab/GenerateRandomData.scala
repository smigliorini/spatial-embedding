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

import edu.ucr.cs.bdlab.beast.generator.SpatialGenerator
import edu.ucr.cs.bdlab.beast.operations.GriddedSummary
import edu.ucr.cs.bdlab.davinci.SingleLevelPlot
import org.apache.hadoop.fs.{Path, PathFilter}
import org.apache.spark.sql.{SaveMode, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}

import java.util.concurrent.TimeoutException

object GenerateRandomData {
  def main(args: Array[String]): Unit = {
    // Create the Spark context
    val conf = new SparkConf
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.setAppName("SpatialGenerator")

    // Set Spark master to local if not already set
    if (!conf.contains("spark.master"))
      conf.setMaster("local[*]")

    val outPath = new Path(if (args.length == 0) "embedding-assorted-datasets" else args(0))

    val spark: SparkSession = SparkSession.builder().config(conf).getOrCreate()
    val sc: SparkContext = spark.sparkContext
    import edu.ucr.cs.bdlab.beast._
    import edu.ucr.cs.bdlab.beast.generator.{BitDistribution, DiagonalDistribution, DistributionType, GaussianDistribution, ParcelDistribution, SierpinskiDistribution, UniformDistribution}
    import edu.ucr.cs.bdlab.beast.geolite.EnvelopeNDLite
    import edu.ucr.cs.bdlab.beast.io.{CSVFeatureWriter, SpatialWriter}

    import scala.concurrent.ExecutionContext.Implicits.global
    import scala.concurrent.duration._
    import scala.concurrent.{Await, Future}
    import scala.util.Random

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
    val numDatasets = 2000
    val datasetsToGenerate = new collection.mutable.ArrayBuffer[Int]()
    datasetsToGenerate ++= 1 to numDatasets
    val datasetsBeingGenerated = new collection.mutable.ArrayBuffer[Future[Int]]()
    val globalMBR = new EnvelopeNDLite(2, 0, 0, 10, 10)
    val concurrency = 32

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
      while (datasetsToGenerate.nonEmpty && datasetsBeingGenerated.size < concurrency) {
        val i = datasetsToGenerate.remove(datasetsToGenerate.size - 1)
        datasetsBeingGenerated.append(Future {
          val random = new Random(i)
          val datasetName = f"dataset-$i%04d"
          val filesystem = outPath.getFileSystem(sc.hadoopConfiguration)
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
          //println(s"${datasetName},${x1},${y1},${x1+mbrWidth},${y1+mbrHeight}")
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
          val dataset: SpatialRDD = generator.generate(cardinality)
          // 1- Write the dataset to the output as a single file
          val datasetFile = new Path(outPath, datasetName+".wkt.bz2")
          if (!filesystem.isFile(datasetFile)) {
            dataset.writeSpatialFile(datasetFile.toString, "wkt",
              Seq(SpatialWriter.CompatibilityMode -> true, CSVFeatureWriter.FieldSeparator -> '\t',
                SpatialWriter.OverwriteOutput -> true))
          }

          // 2- Generate summary
          val summaryPath = new Path(outPath, datasetName+"_summary.csv")
          if (!filesystem.isFile(summaryPath)) {
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
          }
          // 3- Draw an image of it
          SingleLevelPlot.plotFeatures(dataset, 1024, 1024,
            new Path(outPath, datasetName+".png").toString, canvasMBR = globalMBR)
          i
        })
      }
    }

  }

}
