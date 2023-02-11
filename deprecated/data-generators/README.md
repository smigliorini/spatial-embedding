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


## Generate Modified Data
To generate modified data that will hopefully produce more balanced spatial join datasets,
run the following command.
```shell
beast --class edu.ucr.cs.bdlab.GenerateModifiedRandomData target/data-generators-*.jar -o modified-sj-datasets-small -d datasets -s histograms -i /dev/null -gs global-summaries.csv -ds jn_result_3791_balanced.csv
```


## Generate data from a descriptor file
To generate data from a file that contains descriptors, use the following command
```shell
beast --class edu.ucr.cs.bdlab.GenerateFromDescriptors target/data-generators-*.jar --data-descriptor modified-descriptors.json -o modified-sj-datasets-small-rotated -d datasets -s histograms -i /dev/null -gs global-summaries.csv
```