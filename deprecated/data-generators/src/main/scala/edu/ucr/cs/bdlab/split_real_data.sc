import org.apache.spark.SparkContext
val sc: SparkContext = null
import edu.ucr.cs.bdlab.beast._

// Copy/paste starting from here into the Scala shell
import edu.ucr.cs.bdlab.beast.geolite.EnvelopeNDLite
import edu.ucr.cs.bdlab.beast.indexing.CellPartitioner
import edu.ucr.cs.bdlab.beast.generator._

//:paste
val ranges: Array[EnvelopeNDLite] = sc.generateSpatialData
  .distribution(ParcelDistribution)
  .config(SpatialGenerator.Dimensions, 2)
  .config(ParcelDistribution.SplitRange, 0.3)
  .config(ParcelDistribution.Dither, 0)
  .config(SpatialGenerator.Seed, 0)
  .config(SpatialGenerator.AffineMatrix, "360,0,0,180,-180,-90")
  .generate(cardinality=500)
  .map(f => new EnvelopeNDLite(f.getGeometry.getEnvelopeInternal))
  .collect()
//^D

val partitioner = new CellPartitioner(ranges:_*)

sc.spatialFile("parks/data_index", "rtree").spatialPartition(partitioner).saveAsWKTFile("parks_partitioned", 0)
sc.geojsonFile("lakes.geojson.bz2").spatialPartition(partitioner).saveAsWKTFile("lakes_partitioned", 0)