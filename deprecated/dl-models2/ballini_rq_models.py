from tensorflow import keras
from tensorflow.keras import layers


def dense_model(nodes):
    """
    Genera un modello deep dense
        Parameters:
            nodes: Array che indica per ogni livello i numeri di nodi desiderati. Es: [2,3,4] produrrà una rete neurale densa a
            tre livelli con 2^2 nodi nel primo livello, 2^3 nel secondo e 2^4 nel terzo.
    """
    model = keras.Sequential()
    for i in range(0, len(nodes)):
        model.add(layers.Dense(pow(2, nodes[i]), activation="relu"))
    model.add(layers.Dense(1))
    return model


def cnn_GlobalH_4Values(dim, filters1, filters2, nodes1, nodes2, activation):
    """
    Modello CNN che accetta istogramma globale e MBR della range query
        Parameters:
            dim: Dimensione dell'istogramma
            filters1: Numero di filtri per il primo livello convoluzionale
            filters2: Numero di filtri per il secondo livello convoluzionale
            nodes1: Numero di nodi per il primo livello denso
            nodes2: Numero di nodi per il secondo livello denso
            activation: Funzione di attivazione dell'ultimo nodo
    """
    # GLOBAL HISTOGRAMS
    input_ = layers.Input(shape=[(dim ** 2) + 4])
    histogram_gh = input_[:, :dim ** 2]
    histogram_gh = layers.Reshape((dim, dim, 1), input_shape=(dim ** 2, 1))(histogram_gh)

    conv_hist_gh = layers.Conv2D(filters1, (3, 3), activation='relu', padding='same', strides=2)(histogram_gh)
    conv_hist_gh2 = layers.Conv2D(filters2, (3, 3), activation='relu', padding='same', strides=2)(conv_hist_gh)
    conv_hist_gh2 = layers.MaxPooling2D((2, 2))(conv_hist_gh2)
    flatten_hist_gh = layers.Flatten()(conv_hist_gh2)

    # RANGEQUERY
    rq_4 = input_[:, dim ** 2:]

    concat_ = layers.Concatenate()([flatten_hist_gh, rq_4])
    dense1 = layers.Dense(nodes1, activation="relu", )(concat_)
    dense2 = layers.Dense(nodes2, activation="relu")(dense1)
    output_ = keras.layers.Dense(1, activation=activation)(dense2)

    return keras.Model(inputs=[input_], outputs=[output_])


def cnn_GlobalH_RangeH(dim, filters1, filters2, nodes1, nodes2, activation):
    """
    Modello CNN che accetta istogramma globale e istogramma della range query
        Parameters:
            dim: Dimensione dell'istogramma
            filters1: Numero di filtri per il primo livello convoluzionale
            filters2: Numero di filtri per il secondo livello convoluzionale
            nodes1: Numero di nodi per il primo livello denso
            nodes2: Numero di nodi per il secondo livello denso
            activation: Funzione di attivazione dell'ultimo nodo
    """
    # GLOBAL HISTOGRAMS
    input_ = layers.Input(shape=[(dim ** 2) * 2])
    histogram_gh = input_[:, :dim ** 2]
    histogram_gh = layers.Reshape((dim, dim, 1), input_shape=(dim ** 2, 1))(histogram_gh)
    conv_hist_gh = layers.Conv2D(filters1, (3, 3), activation='relu', padding='same', strides=2)(histogram_gh)
    conv_hist_gh2 = layers.Conv2D(filters2, (3, 3), activation='relu', padding='same', strides=2)(conv_hist_gh)
    conv_hist_gh2 = layers.MaxPooling2D((2, 2))(conv_hist_gh2)
    flatten_hist_gh = layers.Flatten()(conv_hist_gh2)

    # GLOBAL RANGEQUERY
    histogram_rq = input_[:, dim ** 2:]
    histogram_rq = layers.Reshape((dim, dim, 1), input_shape=(dim ** 2, 1))(histogram_rq)
    conv_hist_rq = layers.Conv2D(filters1, (3, 3), activation='relu', padding='same', strides=2)(histogram_rq)
    conv_hist_rq2 = layers.Conv2D(filters2, (3, 3), activation='relu', padding='same', strides=2)(conv_hist_rq)
    conv_hist_rq2 = layers.MaxPooling2D((2, 2))(conv_hist_rq2)
    flatten_hist_rq = layers.Flatten()(conv_hist_rq2)

    concat_ = layers.Concatenate()([flatten_hist_gh, flatten_hist_rq])
    dense1 = layers.Dense(nodes1, activation="relu", )(concat_)
    dense2 = layers.Dense(nodes2, activation="relu")(dense1)
    output_ = keras.layers.Dense(1, activation=activation)(dense2)

    return keras.Model(inputs=[input_], outputs=[output_])


def cnn_LocalH_GlobalH_4Values(dim, filters1, filters2, nodes1, nodes2, activation):
    """
    Modello CNN che accetta istogramma globale, istogramma locale e mbr della range query
        Parameters:
            dim: Dimensione dell'istogramma
            filters1: Numero di filtri per il primo livello convoluzionale
            filters2: Numero di filtri per il secondo livello convoluzionale
            nodes1: Numero di nodi per il primo livello denso
            nodes2: Numero di nodi per il secondo livello denso
            activation: Funzione di attivazione dell'ultimo nodo
    """
    # GLOBAL HISTOGRAMS
    input_ = layers.Input(shape=[(dim ** 2) * 2 + 4])
    histogram_gh = input_[:, :dim ** 2]
    histogram_gh = layers.Reshape((dim, dim, 1), input_shape=(dim ** 2, 1))(histogram_gh)
    conv_hist_gh = layers.Conv2D(filters1, (3, 3), activation='relu', padding='same', strides=2)(histogram_gh)
    conv_hist_gh2 = layers.Conv2D(filters2, (3, 3), activation='relu', padding='same', strides=2)(conv_hist_gh)
    conv_hist_gh2 = layers.MaxPooling2D((2, 2))(conv_hist_gh2)
    flatten_hist_gh = layers.Flatten()(conv_hist_gh2)

    # LOCAL HISTOGRAMS
    histogram_lh = input_[:, dim ** 2: (dim ** 2) * 2]
    histogram_lh = layers.Reshape((dim, dim, 1), input_shape=(dim ** 2, 1))(histogram_lh)
    conv_hist_lh = layers.Conv2D(filters1, (3, 3), activation='relu', padding='same', strides=2)(histogram_lh)
    conv_hist_lh2 = layers.Conv2D(filters2, (3, 3), activation='relu', padding='same', strides=2)(conv_hist_lh)
    conv_hist_lh2 = layers.MaxPooling2D((2, 2))(conv_hist_lh2)
    flatten_hist_lh = layers.Flatten()(conv_hist_lh2)

    # RANGEQUERY
    rq_values = input_[:, (dim ** 2) * 2:]

    concat_ = layers.Concatenate()([flatten_hist_gh, flatten_hist_lh, rq_values])
    dense1 = layers.Dense(nodes1, activation="relu", )(concat_)
    dense2 = layers.Dense(nodes2, activation="relu")(dense1)
    output_ = keras.layers.Dense(1, activation=activation)(dense2)

    return keras.Model(inputs=[input_], outputs=[output_])


def cnn_LocalH_GlobalH_RangeH(dim, filters1, filters2, nodes1, nodes2, activation):
    """
    Modello CNN che accetta istogramma globale, istogramma locale e istogramma della range query
        Parameters:
            dim: Dimensione dell'istogramma
            filters1: Numero di filtri per il primo livello convoluzionale
            filters2: Numero di filtri per il secondo livello convoluzionale
            nodes1: Numero di nodi per il primo livello denso
            nodes2: Numero di nodi per il secondo livello denso
            activation: Funzione di attivazione dell'ultimo nodo
    """
    # GLOBAL HISTOGRAMS
    input_ = layers.Input(shape=[(dim ** 2) * 3])
    histogram_gh = input_[:, :dim ** 2]
    histogram_gh = layers.Reshape((dim, dim, 1), input_shape=(dim ** 2, 1))(histogram_gh)
    conv_hist_gh = layers.Conv2D(filters1, (3, 3), activation='relu', padding='same', strides=2)(histogram_gh)
    conv_hist_gh2 = layers.Conv2D(filters2, (3, 3), activation='relu', padding='same', strides=2)(conv_hist_gh)
    conv_hist_gh2 = layers.MaxPooling2D((2, 2))(conv_hist_gh2)
    flatten_hist_gh = layers.Flatten()(conv_hist_gh2)

    # LOCAL HISTOGRAMS
    histogram_lh = input_[:, dim ** 2: (dim ** 2) * 2]
    histogram_lh = layers.Reshape((dim, dim, 1), input_shape=(dim ** 2, 1))(histogram_lh)
    conv_hist_lh = layers.Conv2D(filters1, (3, 3), activation='relu', padding='same', strides=2)(histogram_lh)
    conv_hist_lh2 = layers.Conv2D(filters2, (3, 3), activation='relu', padding='same', strides=2)(conv_hist_lh)
    conv_hist_lh2 = layers.MaxPooling2D((2, 2))(conv_hist_lh2)
    flatten_hist_lh = layers.Flatten()(conv_hist_lh2)

    # RANGEQUERY HISTOGRAMS
    histogram_rq = input_[:, (dim ** 2) * 2:]
    histogram_rq = layers.Reshape((dim, dim, 1), input_shape=(dim ** 2, 1))(histogram_rq)
    conv_hist_rq = layers.Conv2D(filters1, (3, 3), activation='relu', padding='same', strides=2)(histogram_rq)
    conv_hist_rq2 = layers.Conv2D(filters2, (3, 3), activation='relu', padding='same', strides=2)(conv_hist_rq)
    conv_hist_rq2 = layers.MaxPooling2D((2, 2))(conv_hist_rq2)
    flatten_hist_rq = layers.Flatten()(conv_hist_rq2)

    concat_ = layers.Concatenate()([flatten_hist_gh, flatten_hist_lh, flatten_hist_rq])
    dense1 = layers.Dense(nodes1, activation="relu", )(concat_)
    dense2 = layers.Dense(nodes2, activation="relu")(dense1)
    output_ = keras.layers.Dense(1, activation=activation)(dense2)

    return keras.Model(inputs=[input_], outputs=[output_])
