import numpy as np
import tensorflow as tf


def normalize_selectivity(c, arr, max_arr=0):
    """
    Effettua la normalizzazione del dato tramite la formula f(x)=log(1+c*x) e poi applicando una normalizzazione min-max
        Parameters:
            c: Valore per cui viene premoltiplicato arr prima di applicare il logaritmo
            arr: Array da normalizzare
            max_arr: Parametro opzionale che indica il logaritmo del valore massimo che si vuole usare per la
            normalizzazione min-max. Nel caso non venga fornito viene calcolato il massimo dell'array arr.
        Returns:
            Ritorna la tupla che contiene l'array normalizzato e il logaritmo del valore massimo dell'array. Si noti che
            se viene passato il parametro max_arr alla funzione allora la seconda posizione della tupla conterrà proprio il
            parametro max_arr.
    """
    log_arr = np.log1p(c * arr)
    max_arr = np.max(log_arr) if max_arr == 0 else max_arr
    return log_arr / max_arr, max_arr


def denormalize_selectivity(c, arr, max_arr):
    """
        Effettua la denormalizzazione del dato tramite la formula f(x)=exp(x*max_arr-1)/c
            Parameters:
                c: Valore per cui viene è stato premoltiplicato arr prima di applicare il logaritmo nella fase di normalizzazione
                arr: Array normalizzato da denormalizzare
                max_arr: Parametro opzionale che indica il logaritmo del valore massimo che si è usato per la
                normalizzazione min-max nella fase di normalizzazione.
            Returns:
                Ritorna l'array denormalizzato
        """
    log_arr = arr * max_arr
    arr = np.expm1(log_arr) / c
    return arr


def normalize_tvt(train_targets, test_target, validation_target, C):
    """
    Effettua la normalizzazione del training,test e validation set. Per la normalizzazione min-max del validation e del
    test set viene utilizzato il massimo dell training set.
        Parameters:
            train_targets: Array contenente i target del training set
            test_target: Array contenente i target del test set
            validation_target: Array contenente i target del validation set
            C: Valore per cui viene è stato premoltiplicato arr prima di applicare il logaritmo nella fase di normalizzazione
        Returns:
            Ritorna una tupla di 4 valori che contiene i target del training set denormalizzati, i target del test set denormalizzati,
            i target del validation set denormalizzati e il logaritmo del valore massimo di train_targets.
    """
    train_targets, max_train_targets = normalize_selectivity(C, train_targets)
    test_target, _ = normalize_selectivity(C, test_target, max_train_targets)
    validation_target, _ = normalize_selectivity(C, validation_target, max_train_targets)
    return train_targets, test_target, validation_target, max_train_targets


def get_gh(size=64):
    """
    Ritorna l'insieme di 2000 istogrammi globali sottoforma di vettore a cui è stato applicata la funzione
    f(x)=log(1+x). La funzione si aspetta che gli istogrammi si trovino all'interno della cartella embedded e che siano
    disponibili in 3 versioni con i seguenti nomi : 'histograms/global_histograms_128.npy','histograms/global_histograms_64.npy',
    'histograms/global_histograms_32.npy'
        Parameters:
            size: Indica la dimensione della risoluzione desiderata per l'istogramma
        Returns:
            I 2000 istogrammi globali con risoluzione size*size sottoforma di vettore a cui è stata applicata la funzione
            f(x)=log(1+x).
    """
    if size == 128:
        gh = np.load('histograms/global_histograms_128.npy')
    elif size == 64:
        gh = np.load('./histograms/global_histograms_64.npy')
    else:
        gh = np.load('./histograms/global_histograms_32.npy')
    gh = (gh.reshape(2000, gh.shape[1] * gh.shape[2]))
    return np.log1p(gh)


def get_lh(size=64):
    """
    Ritorna l'insieme di 2000 istogrammi locali sottoforma di vettore a cui è stato applicata la funzione
    f(x)=log(1+x). La funzione si aspetta che gli istogrammi si trovino all'interno della cartella embedded e che siano
    disponibili in 3 versioni con i seguenti nomi : 'histograms/local_histograms_128.npy','histograms/local_histograms_64.npy',
    'histograms/local_histograms_32.npy'
        Parameters:
            size: Indica la dimensione della risoluzione desiderata per l'istogramma
        Returns:
            I 2000 istogrammi locali con risoluzione size*size sottoforma di vettore a cui è stata applicata la funzione
            f(x)=log(1+x).
    """
    if size == 128:
        lh = np.load('./histograms/local_histograms_128.npy')
    elif size == 64:
        lh = np.load('./histograms/local_histograms_64.npy')
    else:
        lh = np.load('./histograms/local_histograms_32.npy')

    lh = (lh.reshape(2000, lh.shape[1] * lh.shape[2]))
    return np.log1p(lh)


def get_rh(size=64):
    """
    Ritorna l'insieme di 100000 istogrammi delle range query sottoforma di vettore.
    La funzione si aspetta che gli istogrammi si trovino all'interno della cartella embedded e che siano
    disponibili in 3 versioni con i seguenti nomi : 'histograms/rq_histograms_128.npy','histograms/rq_histograms_64.npy',
    'histograms/rq_histograms_32.npy'
        Parameters:
            size: Indica la dimensione della risoluzione desiderata per l'istogramma
        Returns:
            I 100000 istogrammi delle range query sottoforma di vettore.
    """
    if size == 128:
        rq = np.load('histograms/rq_histograms_128.npy')
    elif size == 64:
        rq = np.load('./histograms/rq_histograms_64.npy')
    else:
        rq = np.load('./histograms/rq_histograms_32.npy')
    return rq.reshape(2000 * 50, rq.shape[1] * rq.shape[2])


def get_rq_data():
    """
    Carica e ritorna il file contenente i dati delle range query. La funzione si aspetta di trovare il file in 'rq_data.npy'.
    Returns:
        Una matrice numpy 100000*8 con i dati delle range query. Essa contiene per ogni range query l'area della range query,
         il valore minimo sull'asse delle X, il valore minimo sull'asse delle Y, il valore massimo sull'asse delle X,
         il valore massimo sull'asse delle Y, la cardinalità del risultato, il numero di mbr test, la cardinalità del dataset
    """
    rq = np.load('rq_data.npy')
    return rq.reshape(2000 * 50, 8)


def get_mbr_Y():
    """
    Carica il file contenente i dati delle range query e ritorna il numero di MBR. La funzione si aspetta di trovare il file
    contenente i dati delle range query in 'rq_data.npy'.
    Returns:
        Il numero di MBR tests
    """
    rqq = np.load('rq_data.npy')
    rqq_new = rqq.reshape(2000 * 50, 8)
    return rqq_new[:, 6]


def get_selectivity_Y():
    """
    Carica il file contenente i dati delle range query e ritorna la selettività delle range query. La funzione si
    aspetta di trovare il file contenente i dati delle range query in 'rq_data.npy'.
        Returns:
            La selettività delle range query
    """
    rqq = np.load('rq_data.npy')
    rqq_new = rqq.reshape(2000 * 50, 8)
    return (rqq_new[:, 5] / rqq_new[:, 7])


def get_cardinality_Y():
    """
    Carica il file contenente i dati delle range query e ritorna la cardinalità delle range query. La funzione si
    aspetta di trovare il file contenente i dati delle range query in 'rq_data.npy'.
        Returns:
            La cardinalità delle range query
    """
    rqq = np.load('rq_data.npy')
    rqq_new = rqq.reshape(2000 * 50, 8)
    return rqq_new[:, 5]


def get_mbr_0_categorical_Y():
    """
    Carica il file contenente i dati delle range query e ritorna un array dove l'i-esima posizione è 0 se la i-esima
    range query ha un numero MBR test uguale a zero, 1 altrimenti. La funzione si
    aspetta di trovare il file contenente i dati delle range query in 'rq_data.npy'.
        Returns:
            Array di valori categorici per la selettività
    """
    rqq = np.load('rq_data.npy')
    rqq_new = rqq.reshape(2000 * 50, 8)
    return 1 * (rqq_new[:, 6] != 0)


def get_selectivity_0_categorical_Y():
    """
    Carica il file contenente i dati delle range query e ritorna un array dove l'i-esima posizione è 0 se la i-esima
    range query ha selettivtià uguale a zero, 1 altrimenti. La funzione si
    aspetta di trovare il file contenente i dati delle range query in 'rq_data.npy'.
        Returns:
            Array di valori categorici per la selettività
    """
    rqq = np.load('rq_data.npy')
    rqq_new = rqq.reshape(2000 * 50, 8)
    return 1 * (rqq_new[:, 5] != 0)


def generate_sets(training_set_index, test_set_index, validation_set_index, y_generator, size=64, rh=False, lh=False):
    """
    Funzione che genera il training, il test e il validation set sia per le X che per le Y.
        Parameters:
            training_set_index: Array di indici che definiscono quali elementi fanno parte del training set
            test_set_index: Array di indici che definiscono quali elementi fanno parte del test set
            validation_set_index: Array di indici che definiscono quali elementi fanno parte del validation set
            y_generator: Funzione che permette di geenrare le Y. (Ad esempio può ritornare un array di cardinalità)
            size: Indica la dimensione desiderata per gli istogrammi.
            rh: Se True utilizza gli istogrammi delle range query. Se False usa invece i 4 valori dell'mbr
            lh: Se True utilizza anche gli istogrammi locali del dataset.
        Returns:
            Una tupla contenente i valori del training,test e  validation set per le X e i valori del training,test e
            validation set per le Y
    """
    gh = get_gh(size)
    y = y_generator()
    x = np.repeat(np.arange(2000), 50)

    gh_max = np.max(gh[x[training_set_index]])
    x_train = gh[x[training_set_index]] / gh_max
    x_test = gh[x[test_set_index]] / gh_max
    x_validation = gh[x[validation_set_index]] / gh_max

    if lh:
        lh = get_lh(size)
        lh_max = np.max(lh[x[training_set_index]])
        lh = lh / lh_max
        x_train = np.concatenate((x_train, lh[x[training_set_index]]), axis=1)
        x_test = np.concatenate((x_test, lh[x[test_set_index]]), axis=1)
        x_validation = np.concatenate((x_validation, lh[x[validation_set_index]]), axis=1)

    if rh:
        rq = get_rh(size)
        rq_max = np.max(rq[training_set_index])
    else:
        rq = get_rq_data()[:, 1:5]
        rq_max = 10
    rq = rq / rq_max

    x_train = np.concatenate((x_train, rq[training_set_index]), axis=1)
    x_test = np.concatenate((x_test, rq[test_set_index]), axis=1)
    x_validation = np.concatenate((x_validation, rq[validation_set_index]), axis=1)
    return x_train, x_test, x_validation, y[training_set_index], y[test_set_index], y[validation_set_index]


def generate_sets_GlobalH_4Values(training_set_index, test_set_index, validation_set_index, y_generator, size=64):
    """
    Genera X e Y per il training il test e il validation set con istogramma globale e 4 valori MBR
    """
    return generate_sets(training_set_index, test_set_index, validation_set_index, y_generator, size, False, False)


def generate_sets_GlobalH_RangeH(training_set_index, test_set_index, validation_set_index, y_generator, size=64):
    """
    Genera X e Y per il training il test e il validation set con istogramma globale e istogramma range query
    """
    return generate_sets(training_set_index, test_set_index, validation_set_index, y_generator, size, True, False)


def generate_sets_LocalH_GlobalH_RangeH(training_set_index, test_set_index, validation_set_index, y_generator, size=64):
    """
    Genera X e Y per il training il test e il validation set con istogramma globale, istogramma locale e istogramma range query
    """
    return generate_sets(training_set_index, test_set_index, validation_set_index, y_generator, size, True, True)


def generate_sets_LocalH_GlobalH_4Values(training_set_index, test_set_index, validation_set_index, y_generator,
                                         size=64):
    """
    Genera X e Y per il training il test e il validation set con istogramma globale,istogramma locale e 4 valori MBR
    """
    return generate_sets(training_set_index, test_set_index, validation_set_index, y_generator, size, False, True)


def rma_metric(t, p):
    """
    Metodo che può essere utilizzato come metrica durante l'allenamento con tensorflow. All'interno dell'algoritmo è presente
    una costante che va cambiata in base alla tipologia di target (cardinalità, selettività, mbr test) e rappresenta il logaritmo
    del valore massimo del target training set.
        Parameters:
            t: Vettore con i target
            p: Vettore con le predizioni
        Returns:
            La metrica RMA
    """
    nz = tf.math.logical_and(t != 0, p != 0)
    rateo = tf.math.divide(tf.divide(tf.math.expm1(tf.math.multiply(tf.boolean_mask(t, nz), 4.946851524809038)), 1000),
                           tf.divide(tf.math.expm1(tf.math.multiply(tf.boolean_mask(p, nz), 4.946851524809038)), 1000))
    l1 = rateo < 1
    m1 = rateo >= 1
    rateo2 = tf.concat([tf.math.reciprocal(tf.boolean_mask(rateo, l1)), tf.boolean_mask(rateo, m1)], 0)
    return tf.math.reduce_mean(rateo2)


def mape_metric(t, p):
    """
    Metodo che può essere utilizzato come metrica durante l'allenamento con tensorflow. All'interno dell'algoritmo è presente
    una costante che va cambiata in base alla tipologia di target (cardinalità, selettività, mbr test) e rappresenta il logaritmo
    del valore massimo del target training set.
        Parameters:
            t: Vettore con i target
            p: Vettore con le predizioni
        Returns:
            La metrica MAPE
    """
    nz = (t != 0)
    den_t = tf.divide(tf.math.expm1(tf.math.multiply(tf.boolean_mask(t, nz), 4.946851524809038)), 1000)
    den_p = tf.divide(tf.math.expm1(tf.math.multiply(tf.boolean_mask(p, nz), 4.946851524809038)), 1000)
    return tf.math.reduce_mean(tf.abs(tf.math.divide(tf.subtract(den_t, den_p), den_t)))
