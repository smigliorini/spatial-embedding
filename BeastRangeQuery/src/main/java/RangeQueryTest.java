import edu.ucr.cs.bdlab.beast.JavaSpatialRDDHelper;
import edu.ucr.cs.bdlab.beast.JavaSpatialSparkContext;
import edu.ucr.cs.bdlab.beast.common.BeastOptions;
import edu.ucr.cs.bdlab.beast.geolite.IFeature;
import edu.ucr.cs.bdlab.beast.io.SpatialReader;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;
import org.locationtech.jts.geom.Envelope;
import org.locationtech.jts.geom.Geometry;
import org.locationtech.jts.geom.GeometryFactory;
import org.apache.spark.util.LongAccumulator;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Stream;

public class RangeQueryTest {
    public static void main(String[] args) {
        // Initialize Spark
        SparkConf conf = new SparkConf().setAppName("Beast Example");
        if (!conf.contains("spark.master"))
            conf.setMaster("local[*]");

        SparkSession sparkSession = SparkSession.builder().config(conf).getOrCreate();
        JavaSparkContext sparkContext = new JavaSparkContext(sparkSession.sparkContext());


        String datasetPath = "", rangeQueriesPath = "", outputPath = "";
        Geometry range = null;
        try {
            /* parsing args */
            datasetPath = args[0];
            rangeQueriesPath = args[1];
            outputPath = args[2];
        } catch (Exception e) {
            System.out.println(e);
        }

        try {
            BufferedReader reader = Files.newBufferedReader(Paths.get(rangeQueriesPath));

            File file = new File(outputPath + "result.csv");
            FileWriter writer = new FileWriter(file);

            writer.write("dataset_numQuery_queryArea,areaInt,cardinality,executionTime(ms),datasetLoading(ms),totalTime(ms),mbrTests\n");


            String line = reader.readLine();
            String datasetLoaded = null;

            GeometryFactory geometryFactory = new GeometryFactory();
            JavaRDD<IFeature> polygons = null;
            long startQuery, executionTime, startDatasetLoading, datasetLoadingTime = 0;

            while ((line = reader.readLine()) != null) {
                // remove unwanted double quotes
                line = line.replaceAll("\"", "");
                // split the line and took correct file
                String[] split = line.split(",");
                String dataset = split[0];

                if (!dataset.equals(datasetLoaded)) {
                    // apro il dataset corretto aggiungendo _grid in fondo
                    System.out.println(datasetPath + "/" + dataset.toLowerCase() + "_grid");
                    startDatasetLoading = System.currentTimeMillis();
                    polygons = SpatialReader.readInput(sparkContext, new BeastOptions().set("separator", ','), datasetPath + "/" + dataset.toLowerCase() + "_grid", "envelope(0,1,2,3)");
                    datasetLoadingTime = System.currentTimeMillis() - startDatasetLoading;
                    datasetLoaded = dataset;

                }
                System.out.println(String.format("Dataset loading time: %d", datasetLoadingTime));
                startQuery = System.currentTimeMillis();
                range = geometryFactory.toGeometry(new Envelope(Double.valueOf(split[3]), Double.valueOf(split[5]), Double.valueOf(split[4]), Double.valueOf(split[6])));
		        LongAccumulator mbr = sparkSession.sparkContext().longAccumulator("MBR");
                JavaRDD<IFeature> matchedPolygons = JavaSpatialRDDHelper.rangeQuery(polygons, range, mbr);
                executionTime = System.currentTimeMillis() - startQuery;
                double cardinality = matchedPolygons.count();
                int numQuery = Integer.valueOf(split[1]);
                String queryArea = split[2];
                String areaInt = split[7];


                System.out.println(String.format("Dataset %s", dataset.toLowerCase()));
                System.out.println(String.format("Query area: %s", queryArea));
                System.out.println(String.format("Num Query: %d", numQuery));
                System.out.println(String.format("Area int: %s", areaInt));
                System.out.println(String.format("MBR tests: %d", mbr.count()));
                System.out.println(String.format("Execution time: %d millis seconds", executionTime));

                writer.write(String.format("%s_%d_%s,%s,%s,%d,%d,%d,%d\n", dataset, numQuery, queryArea.substring(2), areaInt, cardinality, executionTime, datasetLoadingTime, executionTime + datasetLoadingTime, mbr.count()));
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        finally {
            sparkSession.stop();
        }
    }
}
