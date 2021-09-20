# spatial-embedding

The code has the aim to test the configuration of a possible structure for the autoencoder that has to produce the spatial embedding.
It is composed of different files (python modules) as follows:

* myAutoencoder.py: it contains the definition of the network that implements the autoencorder. Here the structure of the network can be modified. Currently it is composed of two parts: the encoder and the decoder and both of them contain only a DENSE layer.
* generate_histogram.py:it contains the code for generating random histograms, normalize them and plot the histogram content or comparing two histograms.
* model0.py: it contains the code for executing one experiment, in particular:
  - It generates histograms by the function create_h(num_instances, dimx, dimly, num_features, type of normalization (0: standardization, 1: norm min-max), type of distribution (0: uniform, 1: diagonal, 2: mix). This function returns 4 arrays: one array of num_instances of local histograms(dimx,dimy,num_features), one array of num_instances of global histograms(dimx,dimy,1) only cardinality is considered here, one array of range query selectivity(dimx,dimy,1) and one array of global histograms representing range query rectangles (dimx,dimy,1). 
  - It splits the array produced in two (training 0.8 and testing 0.2), it generates the network and it trains it. All this by the function auto_encorder(flag_local, flag_global, dimx,dimy,num_feature,local_embeddings_dim, global_embeddings_dim, local histograms, global histograms, range query sel, range query rect). This function produce the trained network, the train and the test set.
  - plot.py: it contains the code for plotting the original and the reconstructed histograms as they were images. Two functions are available: one for histograms with 3 feature, called plot_h, the other one for histograms with 6 features, called plot_h6. Both have the same parameters: plot_h(array of original histograms, array of decoded histograms, start_index, num_histograms_to_show)

In the file run_model0.py there is an example of script that calls the function for generating 2500 histograms 128x128x6 with normalization min-max and distribution mix, then it calls the function for training a network with embedding_dim=1024 and finally it applies the encoding-decoding process to the test_set. At the end the function plot_h6 si called.
