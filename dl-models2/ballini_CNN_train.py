import numpy as np
import tensorflow as tf
from tensorflow import keras
import datetime
import ballini_utils as ut
import ballini_rq_models as rqm
import gc

C = 1000
NUM_EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.0005

MEGABYTES_MEMORY_GPU = 6000
LIMIT_MEMORY_GPU_USAGE = False

SIZE_HISTS = 32
CNN_FILTERS1 = 32
CNN_FILTERS2 = 64
CNN_NODES1 = 512
CNN_NODES2 = 1024

INPUT_TYPES_NAMES = ['G4', 'GH', 'LG4', 'LGH']

# Ci sono casi in cui non si vuole eseguire il training per ognuno dei 4 tipi di input.
# In tal caso si modifica la seguente variabile
input_type_to_be_executed = [0, 1, 2, 3]

TARGET_IS_INT = True    # Indica se il target è di tipo intero (Es: MBR e cardinalità sono interi, la selettività è float)

# Da modificare in base alla tipologia di target che si vuole predire ( selettività, cardinalità, mbr test )
generate_y_function = ut.get_mbr_Y

generate_x_functions = [ut.generate_sets_GlobalH_4Values,
                        ut.generate_sets_GlobalH_RangeH,
                        ut.generate_sets_LocalH_GlobalH_4Values,
                        ut.generate_sets_LocalH_GlobalH_RangeH]

generate_model_functions = [rqm.cnn_GlobalH_4Values,
                            rqm.cnn_GlobalH_RangeH,
                            rqm.cnn_LocalH_GlobalH_4Values,
                            rqm.cnn_LocalH_GlobalH_RangeH]

if(LIMIT_MEMORY_GPU_USAGE):
    max_usage = 0.95 * MEGABYTES_MEMORY_GPU

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_virtual_device_configuration(
              gpus[0],
              [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=max_usage)])
    tf.compat.v1.disable_eager_execution()

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

# Carico gli indici del training, test e validation set
training_set_index = np.load("../dl-models/choosen_impartial_range_queries_index_train_all.npy")
test_set_index = np.load("../dl-models/choosen_impartial_range_queries_index_test_all.npy")
validation_set_index = np.load("../dl-models/choosen_impartial_range_queries_index_validation_all.npy")
print(f"training shape: {training_set_index.shape}\t test shape: {test_set_index.shape}\t validation shape:{validation_set_index.shape}")

for input_type in input_type_to_be_executed:
    print(f"[INFO]: Now working with {INPUT_TYPES_NAMES[input_type]}")
    generate_x_function = generate_x_functions[input_type]
    generate_model_function = generate_model_functions[input_type]

    print("[INFO]: Generating sets")
    train_data, test_data, validation_data, train_targets, test_target, validation_target = generate_x_function(
        training_set_index, test_set_index, validation_set_index, generate_y_function, SIZE_HISTS)

    print("[INFO]: Normalizing sets")
    test_target_den = test_target # Salvo il test target non ancora normalizzato per dopo
    train_targets, test_target, validation_target, max_train_targets = ut.normalize_tvt(
        train_targets, test_target, validation_target, C)

    gc.collect()
    print("[INFO]: Generating and compiling the model")
    model = generate_model_function(SIZE_HISTS, CNN_FILTERS1, CNN_FILTERS2, CNN_NODES1, CNN_NODES2, 'relu')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),loss="mae")

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    print("[INFO]: Fitting the model")
    history = model.fit(train_data, train_targets, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, verbose=1, callbacks=[tensorboard_callback])

    gc.collect()
    prediction = model.predict(test_data, batch_size=BATCH_SIZE)
    prediction = prediction.reshape(prediction.shape[0])
    model = None #Cancello modello per risparmiare sulla memoria

    gc.collect()
    print("[INFO]: Denormalizing prediction")
    prediction_den = ut.denormalize_selectivity(C, prediction, max_train_targets)
    if TARGET_IS_INT:
        prediction_den = np.around(prediction_den).astype(int)

    print("[INFO]: Saving results")
    np.save(f'{INPUT_TYPES_NAMES[input_type]}_test_target', test_target_den)
    np.save(f'{INPUT_TYPES_NAMES[input_type]}_test_prediction', prediction_den)
