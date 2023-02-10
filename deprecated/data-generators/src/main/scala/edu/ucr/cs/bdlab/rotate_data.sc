import org.apache.spark.SparkContext
import org.apache.spark.sql.Row.unapplySeq
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema

val spark: SparkSession = SparkSession.builder().master("local").getOrCreate()
val sc: SparkContext = spark.sparkContext
import edu.ucr.cs.bdlab.beast._

// Start copying from here into Spark shell
import java.io.{FileOutputStream, PrintStream}
import scala.util.matching.Regex
import org.apache.spark.sql.Row
import org.locationtech.jts.geom.Envelope
import java.awt.geom.AffineTransform
import scala.util.Random

def mbr(t: AffineTransform): Envelope = {
  val unitSquare = Array[Double](0, 0, 1, 0, 1, 1, 0, 1)
  t.transform(unitSquare, 0, unitSquare, 0, 4)
  val xs = unitSquare.zipWithIndex.filter(_._2 % 2 == 0).map(_._1)
  val ys = unitSquare.zipWithIndex.filter(_._2 % 2 == 1).map(_._1)
  new Envelope(xs.min, xs.max, ys.min, ys.max)
}

val originalDataSets = spark.read.json("/Volumes/Videos/IdeaProjects/spatial-embedding/deprecated/data-generators/dataset-descriptors.json").collect()
val newData = spark.read.option("delimiter", ",").option("header", true).csv("/Volumes/Videos/IdeaProjects/spatial-embedding/deprecated/data-generators/jn_balanced_2023-01-19-withClasses_rot.csv")
val outputDatasets = new PrintStream(new FileOutputStream("/Volumes/Videos/IdeaProjects/spatial-embedding/deprecated/data-generators/modified-descriptors.json"))
newData.createOrReplaceTempView("newdata")
val newDataPairs = spark.sql("SELECT dataset1, dataset2 FROM newdata").collect.map(r=>(r.getAs[String](0), r.getAs[String](1)))
val rotatedDSExp: Regex = "_r[0-9]+.wkt.bz2".r
for (newDataPair <- newDataPairs) {
  if (rotatedDSExp.findFirstIn(newDataPair._1).isDefined) {
    assert(rotatedDSExp.findFirstIn(newDataPair._2).isDefined)
    val originalDataPair: Seq[Row] = Seq(newDataPair._1, newDataPair._2).map(newData => {
      val originalDataName = rotatedDSExp.replaceFirstIn(newData, "")
      val originalData = originalDataSets.find(r => r.getAs[String]("name") == originalDataName)
      if (originalData.isEmpty)
        throw new RuntimeException(s"${newData} not found")
      originalData.get
    })
    // Rotate the dataset randomly and ensure it fits the global MBR of 0,0,20,20
    val rot = rotatedDSExp.findFirstIn(newDataPair._1).get
    val random = new Random(rot.substring(2, rot.length - ".wkt.bz2".length).toInt)
    val angle = random.nextDouble() * Math.PI
    val matrices = originalDataPair.map(x => new AffineTransform(x.getAs[String]("affineMatrix").split(",").map(_.toDouble)))
    var mbrs = matrices.map(m => mbr(m))
    val center = new Envelope(mbrs(0).centre(), mbrs(1).centre()).centre()
    val rotTransform = AffineTransform.getRotateInstance(angle, center.x, center.y)
    for (m <- matrices)
      m.preConcatenate(rotTransform)
    val globalMBR = new Envelope(0, 10, 0, 10)
    if (!mbrs.forall(globalMBR.contains)) {
      // Tune the transformation to keep it within the globalMBR
      // Tune the transformation to keep it within the globalMBR
      for (i <- 0 to 1) {
        val translate = new AffineTransform()
        if (mbrs(i).getMinX < globalMBR.getMinX)
          translate.translate(globalMBR.getMinX - mbrs(i).getMinX, 0)
        if (mbrs(i).getMinY < globalMBR.getMinY)
          translate.translate(0, globalMBR.getMinY - mbrs(i).getMinY)
        for (m <- matrices)
          m.preConcatenate(translate)
        mbrs = matrices.map(m => mbr(m))
      }
      for (i <- 0 to 1) {
        val scale = new AffineTransform()
        if (mbrs(i).getMaxX > globalMBR.getMaxX)
          scale.scale(globalMBR.getMaxX / mbrs(i).getMaxX, 1.0)
        if (mbrs(i).getMaxY > globalMBR.getMaxY)
          scale.scale(1.0, globalMBR.getMaxY / mbrs(i).getMaxY)
        for (m <- matrices)
          m.preConcatenate(scale)
        mbrs = matrices.map(m => mbr(m))
      }
      assert(mbrs.forall(globalMBR.contains), s"Error with datasets ${newDataPair}, mbrs: ${mbrs}, angle ${angle}")
    }
    // Write the new descriptor to the output
    for (i <- 0 to 1) {
      val originalDescriptor = originalDataPair(i)
      val values: Array[Any] = unapplySeq(originalDescriptor).get.toArray
      val matrixParts = new Array[Double](6)
      matrices(i).getMatrix(matrixParts)
      values(originalDescriptor.fieldIndex("affineMatrix")) = matrixParts.mkString(",")
      val newDescriptor = new GenericRowWithSchema(values, originalDescriptor.schema)
      outputDatasets.println(newDescriptor.json)
    }
  }
}
outputDatasets.close()
