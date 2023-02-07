import com.google.gson.Gson;
import edu.ucr.cs.bdlab.beast.cg.SpatialJoinAlgorithms;
import edu.ucr.cs.bdlab.beast.common.BeastOptions;
import edu.ucr.cs.bdlab.beast.geolite.IFeature;
import edu.ucr.cs.bdlab.beast.io.SpatialReader;
import edu.ucr.cs.bdlab.beast.operations.SpatialJoin;
import org.apache.hadoop.fs.Path;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.util.LongAccumulator;
import scala.Tuple2;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.Date;

/**
 * Execute the spatial joins between the datsets and estract form the logs the statistical information needed.
 */
public class SJMaster {
    private final ArrayList<String> datasets1;
    private final ArrayList<String> datasets_grid1;
    private final ArrayList<String> datasets2;
    private final ArrayList<String> datasets_grid2;
    private Results results ;
    private final SparkSession sparkSession;
    private final JavaSparkContext sparkContext;
    private final ByteArrayOutputStream baos;

    private JavaRDD<IFeature> envelope1;
    private JavaRDD<IFeature> envelope2;
    private JavaRDD<IFeature> envelope1_par;
    private JavaRDD<IFeature> envelope2_par;
    private Path inputDir;

    /**
     *
     * @param pathfile path of the file containing the list of the datasets to be used and their relative partitioned version.
     * @param baos {@link ByteArrayOutputStream} where the StdOut is redirect and that can be used to found the
     *                                          information about the spark execution.
     */
    public SJMaster(String pathfile, ByteArrayOutputStream baos) throws IOException {
        datasets1 = new ArrayList<>();
        datasets_grid1 = new ArrayList<>();
        datasets2 = new ArrayList<>();
        datasets_grid2 = new ArrayList<>();
        this.baos = baos;

        BufferedReader br = new BufferedReader(new FileReader(pathfile));
        String line = br.readLine();
        String[] tmp;
        while (line != null) {
            tmp = line.split(",");
            datasets1.add(tmp[0]);
            datasets_grid1.add(tmp[1]);
            datasets2.add(tmp[2]);
            datasets_grid2.add(tmp[3].replace("\n", ""));
            line = br.readLine();
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

    public void resume(String partialResultsPath){
        Gson gson = new Gson(); // Or use new GsonBuilder().create();
        try {
            this.results = gson.fromJson(new FileReader(partialResultsPath), Results.class);
        } catch (FileNotFoundException e) {
            System.out.println(e.toString());
        }

        int initialNofDatasets = datasets1.size();
        for(String datasetCouple : results.getDatasetCouples()) {
            for (int i = 0; i < datasets1.size(); i++) {
                if (datasetCouple.equals(datasets1.get(i) + "," + datasets2.get(i)) ) {
                    datasets1.remove(i);
                    datasets2.remove(i);
                    datasets_grid1.remove(i);
                    datasets_grid2.remove(i);
                }
            }
        }
        System.out.println("INFO: "+ (initialNofDatasets - datasets1.size())+" couple of datasets has been found in the " +
                "saved results and their execution will be skipped");
    }

    public void setInputDir(Path inputDir) {
        this.inputDir = inputDir;
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
     * @param path Specify the folder where the results will be saved
     * @param safe If set to true it will periodically save the results to avoid losing data especially on long executions.
     * @param algorithmsToUse An array that specify which spatial join algorithms has to be used.
     */
    public void run(String path, boolean safe, boolean[] algorithmsToUse){

        int totalNumCouple = (datasets1.size());
        BeastOptions beastOptions = new BeastOptions().set("skipheader", true);//.set("separator", ',');
        String format = "wkt";
        for (int i = 0; i < totalNumCouple ; i++){
            System.out.println("INFO: Working on the couple n° " + (i+1) +" of " + totalNumCouple+".");

            envelope1 = SpatialReader.readInput(sparkContext, beastOptions, new Path(inputDir, datasets1.get(i)).toString(), format);
            envelope1_par = SpatialReader.readInput(sparkContext, beastOptions, new Path(inputDir, datasets_grid1.get(i)).toString(), format);
            envelope2 = SpatialReader.readInput(sparkContext, beastOptions, new Path(inputDir, datasets2.get(i)).toString(), format);
            envelope2_par = SpatialReader.readInput(sparkContext, beastOptions, new Path(inputDir, datasets_grid2.get(i)).toString(), format);

            results.addEntry(datasets1.get(i) + "," + datasets2.get(i),
                    executeSJ(algorithmsToUse));
            if(safe){
                System.out.println("INFO: Saving the results. Don't kill the process now.");
                results.toCsv(path);
                results.toJson(path);
                System.out.println("INFO: Results saved.");
            }
        }
        System.out.println("INFO: Saving the results. Don't kill the process now.");
        results.toCsv(path);
        results.toJson(path);
        System.out.println("INFO: Results saved.");
    }

    /**
     * Terminate the Spark session
     */
    public void stop(){
        sparkSession.stop();
    }

    /**
     * * Execute the 4 different SpatialJoin between the two datasets.
     * @return A structure containing all the statistical information about the execution of the 4 spatial join.
     * @param algorithmsToUse An array that specify which spatial join algorithms has to be used.
     */
    private SJResult executeSJ(boolean[] algorithmsToUse) {
        SJResult singleResults = new SJResult();
        SpatialJoinAlgorithms.ESJPredicate intersects = SpatialJoinAlgorithms.ESJPredicate.Intersects;
        singleResults.setDataset2Size(envelope2.count());
        singleResults.setDataset1Size(envelope1.count());
        singleResults.setDataset1GridNPartitions(envelope1_par.getNumPartitions());
        singleResults.setDataset2GridNPartitions(envelope2_par.getNumPartitions());
        try {
            baos.reset();
            if(algorithmsToUse[0])
                executeBNLJ(singleResults,intersects);
            if(algorithmsToUse[1])
                executePBSM(singleResults,intersects);
            if(algorithmsToUse[2])
                executeDJ(singleResults,intersects);
            if(algorithmsToUse[3])
                executeREPJ(singleResults,intersects);

        }catch (Exception e){
            System.out.println("An error occurred.");
            System.out.println(e.toString());
        }
        return singleResults;
    }

    private void executeBNLJ(SJResult singleResults, SpatialJoinAlgorithms.ESJPredicate esjPredicate){
        System.out.println("INFO:\tExecuting BNLJ...");
        LongAccumulator mbr = sparkContext.sc().longAccumulator("MBRTests");
        baos.reset();
        JavaPairRDD<IFeature, IFeature> sjResults;

        long start = System.currentTimeMillis();
        sjResults = SpatialJoin.spatialJoin(envelope1, envelope2, esjPredicate,
                SpatialJoinAlgorithms.ESJDistributedAlgorithm.BNLJ, mbr, null, new BeastOptions());
        singleResults.setResultSJSize(sjResults.count());
        singleResults.addJoinResult(JoinAlgorithms.BNLJ, extractSingleSJ(start, mbr.count()));
        baos.reset();
    }

    private void executePBSM(SJResult singleResults, SpatialJoinAlgorithms.ESJPredicate esjPredicate){
        System.out.println("INFO:\tExecuting PBSM...");
        LongAccumulator mbr = sparkContext.sc().longAccumulator("MBRTests");
        baos.reset();
        JavaPairRDD<IFeature, IFeature> sjResults;

        long start = System.currentTimeMillis();
        sjResults = SpatialJoin.spatialJoin(envelope1, envelope2, esjPredicate,
                SpatialJoinAlgorithms.ESJDistributedAlgorithm.PBSM, mbr, null, new BeastOptions());
        singleResults.setResultSJSize(sjResults.count());
        singleResults.addJoinResult(JoinAlgorithms.PBSM, extractSingleSJ(start, mbr.count()));
        baos.reset();
    }
    private void executeDJ(SJResult singleResults, SpatialJoinAlgorithms.ESJPredicate esjPredicate){
        System.out.println("INFO:\tExecuting DJ...");
        LongAccumulator mbr = sparkContext.sc().longAccumulator("MBRTests");
        baos.reset();
        JavaPairRDD<IFeature, IFeature> sjResults;

        long start = System.currentTimeMillis();
        sjResults = SpatialJoin.spatialJoin(envelope1_par, envelope2_par, esjPredicate,
                SpatialJoinAlgorithms.ESJDistributedAlgorithm.DJ, mbr, null, new BeastOptions());
        singleResults.setResultSJSize(sjResults.count());
        singleResults.addJoinResult(JoinAlgorithms.DJ, extractSingleSJ(start, mbr.count()));
        baos.reset();
    }
    private void executeREPJ(SJResult singleResults, SpatialJoinAlgorithms.ESJPredicate esjPredicate){
        System.out.println("INFO:\tExecuting REPJ...");
        LongAccumulator mbr = sparkContext.sc().longAccumulator("MBRTests");
        baos.reset();
        JavaPairRDD<IFeature, IFeature> sjResults;

        long start = System.currentTimeMillis();
        if (envelope1.count() > envelope2.count()) {
            baos.reset();
            sjResults = SpatialJoin.spatialJoin(envelope1_par, envelope2, esjPredicate,
                    SpatialJoinAlgorithms.ESJDistributedAlgorithm.REPJ, mbr, null, new BeastOptions());
        } else {
            baos.reset();
            sjResults = SpatialJoin.spatialJoin(envelope2_par, envelope1, esjPredicate,
                    SpatialJoinAlgorithms.ESJDistributedAlgorithm.REPJ, mbr, null, new BeastOptions());
        }
        singleResults.setResultSJSize(sjResults.count());
        singleResults.addJoinResult(JoinAlgorithms.REPJ, extractSingleSJ(start, mbr.count()));
        baos.reset();
    }

    /**
     * Extracts from the log of the execution the information relative to the execution of a single SJ
     * @param startTime the start time of the join expressed in milliseconds
     * @param count the number of MBR test
     * @return the information relative to the execution of a single SJ
     */
    private AlgorithmResult extractSingleSJ(long startTime,long count){
        long timeElapsed = System.currentTimeMillis() - startTime;
        AlgorithmResult algorithmResult = new AlgorithmResult();
        algorithmResult.setMBRTests(count);
        algorithmResult.setJobsTimes(new ArrayList<>(Arrays.asList(timeElapsed/1000.0) ) );
        algorithmResult.setJobsTimesRelStdDev(0.0);

        algorithmResult.setSizeJoins(new ArrayList<>(Arrays.asList(0L) ) );
        algorithmResult.setSizeJoinsRelStdDev(0);
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

        if (runningTimes.size() == 0)
            errorRetrievingInfo(pattern.pattern(), execOutput);
        return runningTimes;
    }

    /**
     * Extract from the log of the spatial join the size of the joins.
     * For every two partitions of the data to which the join is applied, the size is intended as the multiplication between
     * the size of the two partition.
     * @param execOutput the log of the execution of the spatial join
     * @return An array containing the join sizes.
     */
    private ArrayList<Long> extractSizeJoins(String execOutput) {
        Pattern pattern1 = Pattern.compile("(?<=SpatialJoin: Joining )\\d*");
        Matcher matcher1 = pattern1.matcher(execOutput);
        Pattern pattern2 = Pattern.compile("(?<= x )\\d*(?= records)");
        Matcher matcher2 = pattern2.matcher(execOutput);

        ArrayList<Long> sizeJoins = new ArrayList<>();
        while(matcher1.find() && matcher2.find()){
            long op_size = Long.parseLong(matcher1.group(0)) * Long.parseLong(matcher2.group(0));
            sizeJoins.add(op_size);
        }

        if (sizeJoins.size() == 0)
            errorRetrievingInfo("First regex: "+pattern1.pattern()+
                    "\nSecond regex: "+pattern2.pattern(), execOutput);

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

    private void errorRetrievingInfo(String pattern, String execOutput){
        System.out.println("ERROR: An error occurred while trying to retrieve info with the regex:");
        System.out.println(pattern);
        long name = new Date().getTime();
        System.out.println("The text that couldn't be parsed with the regex can be found in the file "+name+".txt");

        FileWriter file = null;
        BufferedWriter buffer = null;
        try {
            file = new FileWriter(name+".txt");
            buffer = new BufferedWriter(file);
            buffer.write(execOutput);
        } catch (IOException e) {
            System.out.println("ERROR: An error occurred.");
            System.out.println(e.toString());
        } finally {
            try {
                assert buffer != null;
                buffer.close();
                file.close();
            } catch (IOException e) {
                System.out.println("ERROR: An error occurred.");
                System.out.println(e.toString());
            }
        }
    }

}
