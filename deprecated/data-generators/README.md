# Spatial Data Generator
This project generates random spatial data to be used for spatial embeddings.

## Prerequisites
You must have `beast` command-line interface installed.

## How to run
Compile the code into a JAR file using `mvn package`

Generate non-partitioned data and produce a global summary file, individual local summaries, and no images.
Store all output to the `non-partitioned-files` directory.
```shell
beast --class edu.ucr.cs.bdlab.GenerateRandomData target/data-generators-*.jar -d non-partitioned-files -gs global-summaries.csv -i /dev/null 
```