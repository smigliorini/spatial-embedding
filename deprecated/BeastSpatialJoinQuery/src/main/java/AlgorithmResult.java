import java.util.ArrayList;

/**
 * Java object that contains n° of mbr tests, the times of the jobs and the sizes of the joins of a single SpatialJoin.
 */
public class AlgorithmResult {
    public long MBRTests;
    public ArrayList<Double> jobsTimes;
    public double jobsTimesRelStdDev;
    public ArrayList<Long> sizeJoins;
    public double sizeJoinsRelStdDev;

    public AlgorithmResult() {
        this.MBRTests = 0;
        this.jobsTimes = new ArrayList<>();
        this.jobsTimesRelStdDev = 0;
        this.sizeJoins = new ArrayList<>();
        this.sizeJoinsRelStdDev = 0;
    }

    public void setMBRTests(long MBRTests) {
        this.MBRTests = MBRTests;
    }

    public void setJobsTimes(ArrayList<Double> jobsTimes) {
        this.jobsTimes = jobsTimes;
    }

    public void setJobsTimesRelStdDev(double jobsTimesRelStdDev) {
        this.jobsTimesRelStdDev = jobsTimesRelStdDev;
    }

    public void setSizeJoins(ArrayList<Long> sizeJoins) {
        this.sizeJoins = sizeJoins;
    }

    public void setSizeJoinsRelStdDev(double sizeJoinsRelStdDev) {
        this.sizeJoinsRelStdDev = sizeJoinsRelStdDev;
    }
}
