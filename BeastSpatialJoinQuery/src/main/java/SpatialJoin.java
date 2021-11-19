import org.apache.commons.cli.*;
import java.io.*;

public class SpatialJoin {
    public static void main(String[] args) {

        Options options = new Options();
        CommandLineParser parser = new GnuParser();
        options.addOption("s", "safe", false, "Save the partial results to avoid loosing them due to errors at run time");
        options.addOption("b", "bnlj", false, "Execute the spatial joins using the BNLJ algorithm");
        options.addOption("p", "pbsm", false, "Execute the spatial joins using the PBSM algorithm");
        options.addOption("d", "dj", false, "Execute the spatial joins using the DJ algorithm");
        options.addOption("r", "repj", false, "Execute the spatial joins using the REPJ algorithm");
        options.addOption("h", "help", false, "Print this message");
        options.addOption("i", "input", true, "Specify the file containing the list of datasets to use for the spatial join");
        options.addOption("o", "output", true, "Specify the directory that will contain the files with the statistical information regarding the spatial join executions");

        HelpFormatter formatter = new HelpFormatter();
        CommandLine cmd = null;
        try {
            cmd = parser.parse(options, args);
        } catch (ParseException ignored) {
            formatter.printHelp(" ", options);
            System.exit(1);
        }
        if(cmd.hasOption("h")){
            formatter.printHelp(" ", options);
            System.exit(1);
        }

        File f = new File("/tmp/spark-events");
        if (f.mkdir())
            System.err.println("Folder /tmp/spark-events has been created to store the logs of Spark");

        ByteArrayOutputStream baos = new ByteArrayOutputStream(1000000);
        System.setOut(new PrintStream(baos));
        assert cmd != null;

        if( ! new File(cmd.getOptionValue("output", ".")).exists()){
            System.err.println("ERROR:The folder "+cmd.getOptionValue("output", ".")+"/ does not exists.\n" +
                    "The execution will be terminated now");
            System.exit(1);
        }


        SJMaster sjMaster = null;
        try {
            sjMaster = new SJMaster(cmd.getOptionValue("input", "datasets.txt"), baos);
        } catch (FileNotFoundException e) {
            System.err.println("ERROR: Cannot find " + cmd.getOptionValue("input", "datasets.txt")+".\nThe execution will be terminated now");
            System.exit(1);
        }catch (IOException e){
            e.printStackTrace();
            System.err.println("ERROR: The execution will be terminated now");
            System.exit(1);
        }

        boolean[] algorithmToUse;
        if (!cmd.hasOption("b") && !cmd.hasOption("p") && !cmd.hasOption("d") && !cmd.hasOption("r"))
            algorithmToUse = new boolean[]{true,true,true,true};
        else
            algorithmToUse = new boolean[]{cmd.hasOption("b") ,cmd.hasOption("p") ,cmd.hasOption("d") ,cmd.hasOption("r")};

        sjMaster.run(cmd.getOptionValue("output", "."),cmd.hasOption("s"),algorithmToUse);
        sjMaster.stop();

    }

}