# Spatial embeddings project 
 
## Datasets:
### Synthetic datasets:
- small_datasets: 20 MB to 100 MB
- new_small_datasets: 20 MB to 100 MB
- medium datasets: 250 MB to 650 MB
- large datasets: 1GB to 3 GB
- gap datasets: 100MB to 250 MB and 650 to 1GB
- real datasets: lakes and parks, from 1 MB to 4.5 GB 

## Folders organization (each folder contains its own readme file) :
0. summaries: this folder contains the summaries of datasets characteristics, in particular:
	- file mbr_alldatasets.csv: it contains the mbr of the following collections of datasets: small, medium, large and gap
	- file smallDatasets-summaries.csv: it contains the characteristics (including the mbr) of the new_small datasets
	- file lakes_parks-summaries.csv: it contains the characteristics (including the mbr) of the real datasets (lakes and parks) 
1. histograms: this folder contains the histograms (128x128x6) of the synthetic and real datasets in particular:
	- folder small_dataset: it contains csv files representing the histograms of small datasets.
	- folder medium_dataset: it contains csv files representing the histograms of medium datasets.
	- folder gap_dataset: it contains csv files representing the histograms of gap datasets.
	- folder large_dataset: it contains csv files representing the histograms of large datasets. 
	- folder real_dataset: it contains csv files representing the histograms of real datasets.
2. autoEncoders:
	- folder gen_py: it contains the Python code for generating the numpy array to be used as input set for training autoEncoders.
	- folder code_py: it contains the Python code for creating and training different types of autoEncorders.
	- folder generatedTSet: it contains some files representing the numpy array for training autoEncoders.
	- folder trainedModels: it contains some already trained autoEncoders.
3. modelsRQ:
	- folder gen_py: it contains the Python code for generating the numpy array to be used as input set for training M1.
	- folder code_py: it contains the Python code for creating and training different types of networks for the estimation of range query selectivity or #MBRTests (M1).
	- folder generatedTSet: it contains some files representing the numpy array for training models of type M1.
	- folder trainedModels: it contains some already trained model of type M1.
4. modelsSJ:
	- folder gen_py: it contains the Python code for generating the numpy array to be used as input set for training M2.
	- folder code_py: it contains the Python code for creating and training different types of networks for the estimation of spatial join selectivity or #MBRTests (M2).
	- folder generatedTSet: it contains some files representing the numpy array for training models of type M2.
	- folder trainedModels: it contains some already trained model of type M2.
