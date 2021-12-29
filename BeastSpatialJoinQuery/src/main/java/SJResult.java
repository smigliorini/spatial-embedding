import java.util.HashMap;

enum JoinAlgorithms{BNLJ,PBSM,DJ,REPJ}
/*
* It contains the information of the execution of a SpatialJoin with different algorithms.
 */
public class SJResult {



    public long dataset1Size;
    public long dataset2Size;
    public int dataset1GridNPartitions;
    public int dataset2GridNPartitions;
    public long resultSJSize;
    public final HashMap<JoinAlgorithms,AlgorithmResult> SJResultMap;

    public SJResult() {
        SJResultMap = new HashMap<>();
        SJResultMap.put(JoinAlgorithms.BNLJ,new AlgorithmResult());
        SJResultMap.put(JoinAlgorithms.PBSM,new AlgorithmResult());
        SJResultMap.put(JoinAlgorithms.DJ,new AlgorithmResult());
        SJResultMap.put(JoinAlgorithms.REPJ,new AlgorithmResult());
    }

    public void setDataset1Size(long dataset1Size) {
        this.dataset1Size = dataset1Size;
    }

    public void setDataset2Size(long dataset2Size) {
        this.dataset2Size = dataset2Size;
    }

    public void setDataset1GridNPartitions(int dataset1GridNPartitions) {
        this.dataset1GridNPartitions = dataset1GridNPartitions;
    }

    public void setDataset2GridNPartitions(int dataset2GridNPartitions) {
        this.dataset2GridNPartitions = dataset2GridNPartitions;
    }

    public void setResultSJSize(long resultSJSize) {
        this.resultSJSize = resultSJSize;
    }

    public void addJoinResult(JoinAlgorithms joinAlgorithm, AlgorithmResult algorithmResult){
        SJResultMap.put(joinAlgorithm, algorithmResult);
    }
}