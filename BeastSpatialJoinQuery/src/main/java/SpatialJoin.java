import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.io.*;

public class SpatialJoin {
  public static void main(String[] args) {

    File f = new File("/tmp/spark-events");
    if(f.mkdir())
      System.err.println("Folder /tmp/spark-events has been created to store the logs of Spark");

    ByteArrayOutputStream baos = new ByteArrayOutputStream(1000000);
    System.setOut(new PrintStream(baos));

    SJMaster sjMaster = new SJMaster("datasets.txt",baos);
    sjMaster.run();

    Gson gson = new GsonBuilder().setPrettyPrinting().create();

    try {
      FileWriter myWriter = new FileWriter("results.json");
      myWriter.write(gson.toJson(sjMaster.getResults()));
      myWriter.close();
    } catch (IOException e) {
      System.err.println("An error occurred.");
      e.printStackTrace();
    }

    sjMaster.getResults().toCsv();

    sjMaster.stop();

  }

}