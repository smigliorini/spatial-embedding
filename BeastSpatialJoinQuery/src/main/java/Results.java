import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;

public class Results {
    private HashMap<String,SJResult> resultsMap;

    public Results() {
        resultsMap = new HashMap<>();
    }

    public void addEntry(String files, SJResult sjResult){
        resultsMap.put(files,sjResult);
    }

    public void toCsv() {
        FileWriter file;
        try {
            file = new FileWriter("results.csv");

            BufferedWriter buffer = new BufferedWriter(file);

            String header ="dataset1,dataset2, dataset1Cardinality, dataset2Cardinality, dataset1GridNPartitions, dataset2GridNPartitions," +
                    "resultSJSize, " +
                    "BNLJMBRTests, BNLJjobsTimes, BNLJjobsTimesRelStdDev, BNLJsizeJoins, BNLJsizeJoinsRelStdDev,"+
                    "PBSMMBRTests, PBSMjobsTimes, PBSMjobsTimesRelStdDev, PBSMsizeJoins, PBSMsizeJoinsRelStdDev,"+
                    "DJMBRTests, DJjobsTimes, DJjobsTimesRelStdDev, DJsizeJoins, DJsizeJoinsRelStdDev,"+
                    "REPJMBRTests, REPJjobsTimes, REPJjobsTimesRelStdDev, REPJsizeJoins, REPJsizeJoinsRelStdDev\n";
            buffer.write(header);

            for (String datasets : resultsMap.keySet()){
                SJResult sjResult = resultsMap.get(datasets);
                buffer.write(String.format("%s,%d,%d,%d,%d,%d", datasets, sjResult.dataset1Size, sjResult.dataset2Size,
                        sjResult.dataset1GridNPartitions, sjResult.dataset2GridNPartitions, sjResult.resultSJSize));
                for ( JoinAlgorithms algorithm : JoinAlgorithms.values()) {
                    AlgorithmResult sjResultMap = sjResult.SJResultMap.get(algorithm);
                    long totalSizeJoins = 0;
                    for(long sizeJoin : sjResultMap.sizeJoins)
                        totalSizeJoins += sizeJoin;
                    double totalJobsTimes = 0;
                    for(double jobsTime : sjResultMap.jobsTimes)
                        totalJobsTimes += jobsTime;



                    buffer.write(String.format(",%d,%f,%f,%d,%f", sjResultMap.MBRTests, totalJobsTimes,
                            sjResultMap.jobsTimesRelStdDev, totalSizeJoins, sjResultMap.sizeJoinsRelStdDev));
                }
                buffer.newLine();
            }
            buffer.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
