import org.apache.commons.cli.*;
import java.io.*;
import java.util.Arrays;
import java.util.Locale;
import java.util.Scanner;


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
            System.out.println("Folder /tmp/spark-events has been created to store the logs of Spark");

        ByteArrayOutputStream baos = new ByteArrayOutputStream(1000000);
        System.setErr(new PrintStream(baos));
        assert cmd != null;

        if( ! new File(cmd.getOptionValue("output", ".")).exists()){
            System.out.println("ERROR:The folder "+cmd.getOptionValue("output", ".")+"/ does not exists.\n" +
                    "The execution will be terminated now");
            System.exit(1);
        }

        /*
         Check if previous results are present and ask if the user want to resume
         */
        File folder = new File(cmd.getOptionValue("output", "."));
        File [] files = folder.listFiles(new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                return name.endsWith(".json");
            }
        });
        File last = null;
        if (files != null){
            Arrays.sort(files);
            last = files[files.length-1];
            System.out.printf("I found this previous execution:\n%s\nDo you want to resume it? y/n ",last.getName());
            Scanner reader = new Scanner(System.in);
            last = reader.nextLine().toLowerCase().contains("y") ? last : null;
        }

        if (last != null){
            System.out.println("The previous results will be loaded");
        }else{
            System.out.println("The execution will start from scratch");
        }

        SJMaster sjMaster = null;
        try {
            sjMaster = new SJMaster(cmd.getOptionValue("input", "datasets.txt"), baos);
        } catch (FileNotFoundException e) {
            System.err.println("ERROR: Cannot find " + cmd.getOptionValue("input", "datasets.txt")+".\nThe execution will be terminated now");
            System.exit(1);
        }catch (IOException e){
            System.out.println(e.toString());
            System.out.println("ERROR: The execution will be terminated now");
            System.exit(1);
        }

        if(last != null){
            sjMaster.resume(last.getAbsolutePath());
        }

        boolean[] algorithmToUse;
        if (!cmd.hasOption("b") && !cmd.hasOption("p") && !cmd.hasOption("d") && !cmd.hasOption("r"))
            algorithmToUse = new boolean[]{true,true,true,true};
        else
            algorithmToUse = new boolean[]{cmd.hasOption("b") ,cmd.hasOption("p") ,cmd.hasOption("d") ,cmd.hasOption("r")};

        sjMaster.run(cmd.getOptionValue("output", "."),cmd.hasOption("s"),algorithmToUse);
        sjMaster.stop();

        System.exit(0);

    }

}