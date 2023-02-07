import numpy as np
import tensorflow as tf
from tensorflow import keras
import datetime
import ballini_utils as ut
import ballini_rq_models as rqm
import gc
import os

C = 1000
LEARNING_RATE = 0.0005
NUM_EPOCHS = 100
BATCH_SIZE = 64

MEGABYTES_MEMORY_GPU = 6000
LIMIT_MEMORY_GPU_USAGE = False


SIZE_HISTS = 32
CNN_FILTERS1 = 32
CNN_FILTERS2 = 64
CNN_NODES1 = 512
CNN_NODES2 = 1024
NAME_FOLDER = f"Selectivity_After_Cat0{SIZE_HISTS}"

# Ci sono casi in cui non si vuole eseguire il training per ognuno dei 4 tipi di input.
# In tal caso si modifica la seguente variabile
input_type_to_be_executed = [0, 1, 2, 3]

TARGET_IS_INT = True    # Indica se il target è di tipo intero (Es: MBR e cardinalità sono interi, la selettività è float)

# Da modificare in base alla tipologia di target che si vuole predire ( selettività, cardinalità, mbr test )
generate_y_function = ut.get_mbr_Y

INPUT_TYPES_NAMES = ['G4', 'GH', 'LG4', 'LGH']

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

print("[INFO]: Loading indexes")
training_set_index = np.load("../dl-models/choosen_impartial_range_queries_index_train_all.npy")
test_set_index = np.load("../dl-models/choosen_impartial_range_queries_index_test_all.npy")
validation_set_index = np.load("../dl-models/choosen_impartial_range_queries_index_validation_all.npy")

for input_type in input_type_to_be_executed:
    print(f"[INFO]: Now working with {INPUT_TYPES_NAMES[input_type]}")

    model = None
    model_cat0 = None
    gc.collect()

    generate_x_function = generate_x_functions[input_type]
    generate_model_function = generate_model_functions[input_type]

    print("[INFO]: Generating sets")
    train_data, test_data, validation_data, train_targets, test_targets, validation_targets = generate_x_function(
        training_set_index, test_set_index, validation_set_index, generate_y_function, SIZE_HISTS)

    ############## CATEGORICAL ZERO / NOT ZERO ##############
    print("[INFO]: Training zeros / non zeros")
    y_train_not_zero = train_targets != 0
    y_test_not_zero = test_targets != 0
    y_validation_not_zero = validation_targets != 0
    train_targets_cat0 = 1 * (y_train_not_zero)
    test_targets_cat0 = 1 * (y_test_not_zero)
    validation_targets_cat0 = 1 * (y_validation_not_zero)

    print("[INFO]: Generating and compiling the model")
    model_cat0 = generate_model_function(SIZE_HISTS, CNN_FILTERS1, CNN_FILTERS2, CNN_NODES1, CNN_NODES2, 'sigmoid')

    model_cat0.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                       loss="binary_crossentropy",
                       metrics=["accuracy"])

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    print("[INFO]: Fitting the model")
    history = model_cat0.fit(train_data, train_targets_cat0, epochs=int(NUM_EPOCHS / 3), batch_size=BATCH_SIZE,
                             verbose=1,
                             validation_data=(validation_data, validation_targets_cat0), callbacks=[tensorboard_callback])

    test_prediction_cat0 = model_cat0.predict(test_data, batch_size=BATCH_SIZE)
    test_prediction_cat0 = 1 * (
                test_prediction_cat0.reshape(test_prediction_cat0.shape[0]) >= .5)  # 1 = not zero, 0 = zero
    model_cat0 = None #Cancello modello per risparmiare sulla memoria

    gc.collect()

    ############## TRAIN CARDINALITY ##############
    print("[INFO]: Training cardinality")

    print("[INFO]: Normalizing sets")
    test_targets_den = test_targets.copy() # Salvo il test target non ancora normalizzato per dopo
    train_targets, test_targets, validation_targets, max_train_targets = ut.normalize_tvt(
        train_targets, test_targets, validation_targets, C)

    print("[INFO]: Generating and compiling the model")
    model = generate_model_function(SIZE_HISTS, CNN_FILTERS1, CNN_FILTERS2, CNN_NODES1, CNN_NODES2, 'relu')  # dense_model(nodes)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss="mae")  # metrics=[ut.mape_metric, ut.rma_metric] learning_rate=0.0005

    print("[INFO]: Fitting the model")
    history = model.fit(train_data[y_train_not_zero], train_targets[y_train_not_zero], epochs=NUM_EPOCHS,
                        batch_size=BATCH_SIZE,
                        verbose=1, validation_data=(
        validation_data[y_validation_not_zero], validation_targets[y_validation_not_zero]))

    gc.collect()
    ########## NOW LET'S OBTAIN THE RESULTS ON THE TEST SET ###########

    test_prediction = model.predict(test_data, batch_size=BATCH_SIZE)
    model = None
    test_prediction = test_prediction.reshape(test_prediction.shape[0])

    print("[INFO]: Denormalizing prediction")
    test_prediction_den = ut.denormalize_selectivity(C, test_prediction, max_train_targets)
    if TARGET_IS_INT:
        test_prediction_den = np.around(test_prediction_den).astype(int)

    raw_test_prediction = test_prediction_den.copy()

    test_prediction_den[test_prediction_cat0 == 0] = 0
    print(f"""[INFO]: After passing to the three models the test set this are the results:
    Number of 0 predicted by the categorical classifier: {np.sum(test_prediction_cat0 == 0)}
    """)

    print("[INFO]: Saving results")
    saving_path = os.path.join(NAME_FOLDER + INPUT_TYPES_NAMES[input_type])
    os.makedirs(saving_path)

    np.save(os.path.join(saving_path, f'{INPUT_TYPES_NAMES[input_type]}_test_target'), test_targets_den)
    np.save(os.path.join(saving_path, f'{INPUT_TYPES_NAMES[input_type]}_test_prediction'), test_prediction)

    np.save(os.path.join(saving_path, f'{INPUT_TYPES_NAMES[input_type]}_cat_0_test_target'), test_targets_cat0)
    np.save(os.path.join(saving_path, f'{INPUT_TYPES_NAMES[input_type]}_cat_0_test_prediction'), test_prediction_cat0)

    np.save(os.path.join(saving_path, f'{INPUT_TYPES_NAMES[input_type]}_raw_test_target'), test_targets_den)
    np.save(os.path.join(saving_path, f'{INPUT_TYPES_NAMES[input_type]}_raw_test_prediction'), raw_test_prediction)
