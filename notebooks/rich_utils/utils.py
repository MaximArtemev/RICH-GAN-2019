from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
import numpy as np
import pandas as pd
import tensorflow as tf
import os.path
import re

names_labels_correspondence = {
    "electron": 0,
    "muon": 1,
    # TODO(kazeevn) this is a hacky hack
    "pion2": 2,
    "kaon2": 3,
    "proton": 4}

dll_columns = ['RichDLLe', 'RichDLLk', 'RichDLLmu', 'RichDLLp', 'RichDLLbt']
feature_columns = [ 'Brunel_P', 'Brunel_ETA', 'nTracks_Brunel']
weight_column = 'probe_sWeight'
# placing y-x-weights
ONE_AND_TRUE_COLUMNS_ORDER = ['RichDLLbt', 'RichDLLe', 'RichDLLk', 'RichDLLmu', 'RichDLLp',
                              'Brunel_P', 'Brunel_ETA', 'nTracks_Brunel', 'charge',
                              'magnet', 'type_0', 'type_1', 'type_2', 'type_3', 'type_4', 'probe_sWeight']
                     
y_count = len(dll_columns)
TEST_SIZE = 0.5

NAME_PARSER = re.compile(r'(?P<type>\w+)_(?P<charge>[+-])_(?P<magnet>\w+)_(?P<year>\d+)_\.csv')

def load_file(file_name, dtype=None):
    """Loads data from file. Sets particle type, charge and magnet
    polarity from the file name"""
    parsed_name = NAME_PARSER.match(os.path.basename(file_name))
    if dtype is None:
        column_type = int
    else:
        column_type = dtype
    columns_to_read = dll_columns+feature_columns+[weight_column]
    if parsed_name.group("type") == "electron":
        print("WARNING! Writing nTracks into nTracks_Brunel for electrons.")
        columns_to_read.remove("nTracks_Brunel")
        columns_to_read.append("nTracks")
        data = pd.read_csv(file_name, delimiter='\t',
                           usecols=columns_to_read,
                           dtype=dtype).rename(
                               columns={"nTracks": "nTracks_Brunel"},
                               copy=False)
        # TODO(kazeevn) can we get the charge?
        data["charge"] = column_type(0)
    else:
        data = pd.read_csv(file_name, delimiter='\t',
                           usecols=columns_to_read,
                           dtype=dtype)
        data["charge"] = column_type((parsed_name.group("charge") == "+")*2 - 1)
    data["magnet"] = column_type((parsed_name.group("magnet") == "up")*2 - 1)
    true_type = names_labels_correspondence[parsed_name.group("type")]
    for type_ in names_labels_correspondence.values():
        data["type_%i" % type_] = column_type(true_type == type_)
    return data


def split(data):
    data_train, data_val = train_test_split(data, test_size=TEST_SIZE, random_state=42)
    data_val, data_test = train_test_split(data_val, test_size=TEST_SIZE, random_state=1812)
    return data_train.reset_index(drop=True), data_val.reset_index(drop=True), data_test.reset_index(drop=True)


def get_tf_datasets(datasets, batch_size):
    shuffler = tf.contrib.data.shuffle_and_repeat(datasets[0].shape[0])
    suffled_ds = shuffler(tf.data.Dataset.from_tensor_slices(datasets))
    return suffled_ds.batch(batch_size).prefetch(1).make_one_shot_iterator().get_next()


def scale_pandas(dataframe, scaler):
    return pd.DataFrame(scaler.transform(dataframe.values), columns=dataframe.columns)


def get_full_dataset(path, dtype=np.float32):
    df_list = []
    for file_name in os.listdir(path):
        if file_name.endswith(".csv"):
            df_list.append(load_file(os.path.join(path, file_name), dtype))
    data_full = pd.concat(df_list, ignore_index=True, join="inner", copy=False)[ONE_AND_TRUE_COLUMNS_ORDER]
    # Since we'll be doing quite some parameter search,
    # we'll ignore the test for now
    data_train, data_val, _ = split(data_full)
    columns_to_rescale = dll_columns + feature_columns
    scaler = QuantileTransformer(output_distribution="normal",
                                 n_quantiles=int(1e5),
                                 subsample=int(1e10),
                                 copy=False)
    # TODO(kazeev) does it copy?
    print("It will now print a warning, but still work. Most likely due to"
          " a copy of data_full made by train_test_split, but feel free to"
          " investigate.")
    data_train.loc[:, columns_to_rescale] = scaler.fit_transform(
        data_train.loc[:, columns_to_rescale].values).astype(dtype)
    data_val.loc[:, columns_to_rescale] = scaler.transform(
        data_val.loc[:, columns_to_rescale].values).astype(dtype)

    return data_train, data_val, scaler
