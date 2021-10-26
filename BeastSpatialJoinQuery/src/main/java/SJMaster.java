import edu.ucr.cs.bdlab.beast.cg.SpatialJoinAlgorithms;
import edu.ucr.cs.bdlab.beast.common.BeastOptions;
import edu.ucr.cs.bdlab.beast.geolite.IFeature;
import edu.ucr.cs.bdlab.beast.io.SpatialReader;
import edu.ucr.cs.bdlab.beast.operations.SpatialJoin;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.util.LongAccumulator;
import scala.Tuple2;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Execute the spatial joins between the datsets and estract form the logs the statistical information needed.
 */
public class SJMaster {
    private final ArrayList<String> datasets;
    private final ArrayList<String> datasets_grid;
    private Results results ;
    private final SparkSession sparkSession;
    private JavaSparkContext sparkContext;
    private final ByteArrayOutputStream baos;

    /**
     *
     * @param pathfile path of the file containing the list of the datasets to be used and their relative partitioned version.
     * @param baos {@link ByteArrayOutputStream} where the StdOut is redirect and that can be used to found the
     *                                          information about the spark execution.
     */
    public SJMaster(String pathfile, ByteArrayOutputStream baos) {
        datasets = new ArrayList<>();
        datasets_grid = new ArrayList<>();
        this.baos = baos;
        try {
            BufferedReader br = new BufferedReader(new FileReader(pathfile));
            String line = br.readLine();
            while (line != null) {
                datasets.add(line.split(",")[0]);
                datasets_grid.add(line.split(",")[1].replace("\n",""));
                line = br.readLine();
            }
        }catch (Exception e){
            e.printStackTrace();
        }

        SparkConf conf = new SparkConf().setAppName("Beast Example");
        if (!conf.contains("spark.master"))
            conf.setMaster("local[*]");
        conf.set("org.apache.spark.serializer.JavaSerializer", "org.apache.spark.serializer.KryoSerializer");
        conf.set("spark.submit.deployMode","client");
        conf.set("spark.eventLog.enabled","true"); // Save log in /tmp/spark-events
        conf.set("spark.files.overwrite", "true");
        sparkSession = SparkSession.builder().config(conf).getOrCreate();
        sparkContext = new JavaSparkContext(sparkSession.sparkContext());

        // Split file in partition of 4Mb during the reading phase
        sparkContext.hadoopConfiguration().setLong("mapred.max.split.size",
                new BeastOptions().getSizeAsBytes("JoinWorkloadUnit", "4m"));
        sparkContext.setLogLevel("INFO");

        results = new Results();
    }

    /**
     *  Return the results of the executions of the spatial joins
     */
    public Results getResults() {
        return results;
    }

    /**
     * Execute the spatial join between every couple of datasets and store the results.
     * The results can be retrieved using the method {@link #getResults()}
     */
    public void run(){
        BeastOptions beastOptions = new BeastOptions().set("separator", ',');
        String format = "envelope(0,1,2,3)";
        for (int i = 0; i < datasets.size() - 1 ; i++){
            JavaRDD<IFeature> envelope1 = SpatialReader.readInput(sparkContext, beastOptions, datasets.get(i), format);
            JavaRDD<IFeature> envelope1_par = SpatialReader.readInput(sparkContext, beastOptions, datasets_grid.get(i), format);
            for (int j = i+1; j < datasets.size(); j++){
                JavaRDD<IFeature> envelope2 = SpatialReader.readInput(sparkContext, beastOptions, datasets.get(j), format);
                JavaRDD<IFeature> envelope2_par = SpatialReader.readInput(sparkContext, beastOptions, datasets_grid.get(j), format);
                results.addEntry(datasets.get(i) + "," + datasets.get(j),
                        executeSJ(envelope1,envelope2,envelope1_par,envelope2_par));
            }
        }
    }


    /**
     * Terminate the Spark session
     */
    public void stop(){
        sparkSession.stop();
    }

    /**
     * * Execute the 4 different SpatialJoin between the two datasets.
     * @param envelope1 The first dataset
     * @param envelope2 The second dataset
     * @param envelope1_par The first dataset partitioned
     * @param envelope2_par The second dataset partitioned
     * @return A structure containing all the statistical information about the execution of the 4 spatial join.
     */
    private SJResult executeSJ(JavaRDD<IFeature> envelope1, JavaRDD<IFeature> envelope2, JavaRDD<IFeature> envelope1_par,JavaRDD<IFeature> envelope2_par) {
        SJResult singleResults = new SJResult();
        SpatialJoinAlgorithms.ESJPredicate intersects = SpatialJoinAlgorithms.ESJPredicate.Intersects;
        try {
            singleResults.setDataset2Size(envelope2.count());
            singleResults.setDataset1Size(envelope1.count());
            singleResults.setDataset1GridNPartitions(envelope1_par.getNumPartitions());
            singleResults.setDataset2GridNPartitions(envelope2_par.getNumPartitions());
            LongAccumulator mbr = sparkContext.sc().longAccumulator("MBRTests");
            baos.reset();
            RDD<Tuple2<IFeature, IFeature>> sjResults;
            sjResults = SpatialJoin.spatialJoin(envelope1.rdd(), envelope2.rdd(), intersects,
                    SpatialJoinAlgorithms.ESJDistributedAlgorithm.BNLJ, mbr, new BeastOptions());
            singleResults.setResultSJSize(sjResults.count());
            singleResults.addJoinResult(JoinAlgorithms.BNLJ, extractSingleSJ(baos.toString(), mbr.count()));
            baos.reset();
            mbr.reset();

            sjResults = SpatialJoin.spatialJoin(envelope1.rdd(), envelope2.rdd(), intersects,
                    SpatialJoinAlgorithms.ESJDistributedAlgorithm.PBSM, mbr, new BeastOptions());
            System.err.println(sjResults.first());
            singleResults.addJoinResult(JoinAlgorithms.PBSM, extractSingleSJ(baos.toString(), mbr.count()));
            baos.reset();
            mbr.reset();

            sjResults = SpatialJoin.spatialJoin(envelope1_par.rdd(), envelope2_par.rdd(), intersects,
                    SpatialJoinAlgorithms.ESJDistributedAlgorithm.DJ, mbr, new BeastOptions());
            System.err.println(sjResults.first());
            singleResults.addJoinResult(JoinAlgorithms.DJ, extractSingleSJ(baos.toString(), mbr.count()));
            mbr.reset();

            if (envelope1.count() > envelope2.count()) {
                baos.reset();
                sjResults = SpatialJoin.spatialJoin(envelope1_par.rdd(), envelope2.rdd(), intersects,
                        SpatialJoinAlgorithms.ESJDistributedAlgorithm.REPJ, mbr, new BeastOptions());
            } else {
                baos.reset();
                sjResults = SpatialJoin.spatialJoin(envelope2_par.rdd(), envelope1.rdd(), intersects,
                        SpatialJoinAlgorithms.ESJDistributedAlgorithm.REPJ, mbr, new BeastOptions());
            }
            System.err.println(sjResults.first());
            singleResults.addJoinResult(JoinAlgorithms.REPJ, extractSingleSJ(baos.toString(), mbr.count()));
            baos.reset();
            mbr.reset();
        }catch (Exception e){
            e.printStackTrace();
        }
        return singleResults;
    }

    /**
     * Extracts from the log of the execution the information relative to the execution of a single SJ
     * @param execOutput the log of the execution of the spatial join
     * @param count the number of MBR test
     * @return the information relative to the execution of a single SJ
     */
    private AlgorithmResult extractSingleSJ(String execOutput,long count){
        AlgorithmResult algorithmResult = new AlgorithmResult();
        algorithmResult.setMBRTests(count);
        ArrayList<Double> jobsTime = extractRunningTimes(execOutput);
        algorithmResult.setJobsTimes(jobsTime);
        algorithmResult.setJobsTimesRelStdDev(RSDTimes(jobsTime));

        ArrayList<Long> sizeJoins = extractSizeJoins(execOutput);
        algorithmResult.setSizeJoins(sizeJoins);
        algorithmResult.setSizeJoinsRelStdDev(RSDSizeJoins(sizeJoins));
        return  algorithmResult;
    }

    /**
     * Extract from the log of the spatial join the time of execution of each job.
     * @param execOutput the log of the execution of the spatial join
     * @return An array containing the times.
     */
    private ArrayList<Double> extractRunningTimes(String execOutput) {
        Pattern pattern = Pattern.compile("(?<=took )\\d*.\\d*");
        Matcher matcher = pattern.matcher(execOutput);
        ArrayList<Double> runningTimes = new ArrayList<>();

        while(matcher.find()){
            runningTimes.add(Double.valueOf(matcher.group(0)));
        }
        return runningTimes;
    }

    /**
     * Extract from the log of the spatial join the size of the joins.
     * For every two partitons of the data to which the join is applied, the size is intended as the multiplication between
     * the size of the two partition.
     * @param execOutput the log of the execution of the spatial join
     * @return An array containing the join sizes.
     */
    private ArrayList<Long> extractSizeJoins(String execOutput) {
        Pattern pattern1 = Pattern.compile("(?<=SpatialJoin  - Joining )\\d*");
        Matcher matcher1 = pattern1.matcher(execOutput);
        Pattern pattern2 = Pattern.compile("(?<= x )\\d*(?= records)");
        Matcher matcher2 = pattern2.matcher(execOutput);

        ArrayList<Long> sizeJoins = new ArrayList<>();
        while(matcher1.find() && matcher2.find()){
            long op_size = Long.parseLong(matcher1.group(0)) * Long.parseLong(matcher2.group(0));
            sizeJoins.add(op_size);
        }
        return sizeJoins;
    }

    /**
     * Extract the relative standard variation from the size of the joins
     * @param values sizes of the joins
     * @return The relative standard variation of the joins' sizes
     */
    private Double RSDSizeJoins(ArrayList<Long> values){
        double mean = 0.0;
        double deviation = 0.0;
        int num_joins = values.size();
        for (Long sizeJoin : values){
            mean += (sizeJoin*1.0)/num_joins;
        }
        for (Long sizeJoin : values){
            deviation += (Math.pow(sizeJoin-mean,2))/num_joins;
        }
        deviation = Math.sqrt(deviation);
        return deviation/mean;
    }

    /**
     * Extract the relative standard variation from execution times of the jobs
     * @param values execution times of the jobs
     * @return The relative standard variation of the jobs' times
     */
    private Double RSDTimes(ArrayList<Double> values){
        double mean = 0.0;
        double deviation = 0.0;
        int num_joins = values.size();
        for (Double sizeJoin : values){
            mean += sizeJoin/num_joins;
        }
        for (Double sizeJoin : values){
            deviation += (Math.pow(sizeJoin-mean,2))/num_joins;
        }
        deviation = Math.sqrt(deviation);
        return deviation/mean;
    }
}
