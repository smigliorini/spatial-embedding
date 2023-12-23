import edu.ucr.cs.bdlab.beast.generator.UniformDistribution
import edu.ucr.cs.bdlab.beast.geolite.EnvelopeNDLite
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
object GenerateDescriptors {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    if (!conf.contains("spark.master"))
      conf.setMaster("local[*]")
    val spark: SparkSession = SparkSession.builder().config(conf).getOrCreate()
    val sc: SparkContext = spark.sparkContext
    import edu.ucr.cs.bdlab.beast._

    val baseDir = "/Users/eldawy/IdeaProjects/spatial-embedding/deprecated/BeastSpatialJoinQuery/"
    val dataPairs = spark.read.option("separator", ",")
      .option("header", true)
      .option("inferschema", true)
      .csv(baseDir + "datasets_new_join_cost_model-synth-pairs.csv")
      .collect()

    for (dataPair <- dataPairs) {
      val datasets = Array(
        (dataPair.getAs[String]("D1"), dataPair.getAs[Double]("avgArea D1"), dataPair.getAs[Int]("size D1 (in MB)"), dataPair.getAs[String]("MBR1")),
        (dataPair.getAs[String]("D2"), dataPair.getAs[Double]("avgArea D2"), dataPair.getAs[Int]("size D2 (in MB)"), dataPair.getAs[String]("MBR2")),
      )
      datasets.foreach(data => {
        val name = data._1
        val avgArea = data._2
        val cardinality: Long = (data._3 * 1024.0 * 1024 / 64).toLong
        val mbrAr = data._4.split(",").map(_.toDouble)
        val mbr = new EnvelopeNDLite(mbrAr.slice(0, 2), mbrAr.slice(2, 4))
        val maxWidth = 2*Math.sqrt(avgArea) / mbr.getSideLength(0)
        val maxHeight = 2*Math.sqrt(avgArea) / mbr.getSideLength(1)
        val generator = sc.generateSpatialData.mbr(mbr)
          .makeBoxes(maxWidth, maxHeight)
          .distribution(UniformDistribution)
        val pattern = "([AB])([0-9]{1,3})".r
        val (letter: String, number: Int) = name match {
          case pattern(letter, number) => (letter, number.toInt)
          case _ => null
        }
        val opts: Map[String, String] = generator.config ++
          Seq("distribution" -> "uniform", "cardinality" -> cardinality.toString, "seed" -> (number + letter(0) - 'A').toString, "name" -> name)

        import com.fasterxml.jackson.databind.ObjectMapper
        import com.fasterxml.jackson.module.scala.DefaultScalaModule
        import com.fasterxml.jackson.module.scala.ScalaObjectMapper

        val json = new ObjectMapper with ScalaObjectMapper
        val str = json.registerModule(DefaultScalaModule)
          .writeValueAsString(opts)

        println(str)
      })
    }

  }
}
