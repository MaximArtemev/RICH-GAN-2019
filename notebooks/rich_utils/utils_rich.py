from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, QuantileTransformer
import numpy as np
import pandas as pd
import tensorflow as tf

data_dir = '/home/mrartemev/storage/Data/Lambda/Rich/data_calibsample/'

datasets_kaon = [data_dir + 'kaon' + final_str + '_.csv' for final_str in ['_-_down_2016',
                                                                           '_-_up_2016',
                                                                           '_+_down_2016',
                                                                           '_+_up_2016']]
                 
datasets_pion = [data_dir + 'pion' + final_str + '_.csv' for final_str in ['_-_down_2016',
                                                                           '_-_up_2016',
                                                                           '_+_down_2016',
                                                                           '_+_up_2016']]
 
datasets_proton = [data_dir + 'proton' + final_str + '_.csv' for final_str in ['_-_down_2016',
                                                                           '_-_up_2016',
                                                                           '_+_down_2016',
                                                                           '_+_up_2016']]

datasets_muon = [data_dir + 'muon' + final_str + '_.csv' for final_str in ['_-_down_2016',
                                                                           '_-_up_2016',
                                                                           '_+_down_2016',
                                                                           '_+_up_2016']]


dll_columns = ['RichDLLe', 'RichDLLk', 'RichDLLmu', 'RichDLLp', 'RichDLLbt']
raw_feature_columns = [ 'Brunel_P', 'Brunel_ETA', 'nTracks_Brunel' ]
weight_col = 'probe_sWeight'

#non_type_features = [
#    'particle_one_energy', 'particle_two_energy',
#    'particle_one_eta', 'particle_two_eta', 'particle_one_x', 'particle_two_x']
                     
                     
y_count = len(dll_columns)
TEST_SIZE = 0.5
#ENERGY_CUT = 2.5

def load_and_cut(file_name):
    data = pd.read_csv(file_name, delimiter='\t')
#    data = data[(data.particle_one_energy > ENERGY_CUT)]
#    assert not ((data[['support_electron', 'support_kaon', 'support_muon',
#                       'support_proton', 'support_pion']] == 0).any()).any()
    return data[dll_columns+raw_feature_columns+[weight_col]]

def load_and_merge_and_cut(filename_list):
    return pd.concat([load_and_cut(fname) for fname in filename_list], axis=0, ignore_index=True)

#def load_and_preprocess_dataset(file_name):
#    data = load_and_cut(file_name)
#    data = pd.get_dummies(data[dll_columns+raw_feature_columns], prefix=['particle_one_type'],
#                          columns=['particle_one_type'], drop_first=True)
#    assert (data.columns == ['dll_electron', 'dll_kaon', 'dll_muon', 'dll_proton', 'dll_bt',
#                             'particle_one_energy', 'particle_two_energy', 'particle_one_eta',
#                             'particle_two_eta', 'particle_one_x', 'particle_two_x',
#                             'particle_one_type_1',
#                             'particle_one_type_2', 'particle_one_type_3', 
#                             'particle_one_type_4']).all()
#    return data


def split(data):
    data_train, data_val = train_test_split(data, test_size=TEST_SIZE, random_state=42)
    data_val, data_test = train_test_split(data_val, test_size=TEST_SIZE, random_state=1812)
    return data_train.reset_index(drop=True), \
           data_val  .reset_index(drop=True), \
           data_test .reset_index(drop=True)

#def split_and_scale(data):
#    data_train, data_val = train_test_split(data, test_size=TEST_SIZE, random_state=42)
#    scaler = RobustScaler().fit(data_train)
#    # pandas...
#    data_train = pd.DataFrame(scaler.transform(data_train.values),
#                              columns=data_train.columns)
#    data_val = pd.DataFrame(scaler.transform(data_val.values),
#                            columns=data_val.columns)
#    data_val, data_test = train_test_split(data_val, test_size=TEST_SIZE, random_state=1812)
#    return data_train, data_val, data_test, scaler

def get_tf_dataset(dataset, batch_size):
    shuffler = tf.contrib.data.shuffle_and_repeat(dataset.shape[0])
    suffled_ds = shuffler(tf.data.Dataset.from_tensor_slices(dataset))
    return suffled_ds.batch(batch_size).prefetch(1).make_one_shot_iterator().get_next()

#def select_by_cuts(data, variable_cut):
#    selected_data = np.ones(len(data), dtype=np.bool)
#    for variable, cut in variable_cut.items():
#        selected_data &= (data[variable] < cut[1]) & (data[variable] >= cut[0])
#    return data.loc[selected_data]

def scale_pandas(dataframe, scaler):
    return pd.DataFrame(scaler.transform(dataframe.values), columns=dataframe.columns)

def get_merged_typed_dataset(particle_type, dtype=None, log=False):
    if particle_type == 'kaon':
        file_list = datasets_kaon
    elif particle_type == 'pion':
        file_list = datasets_pion
    elif particle_type == 'proton':
        file_list = datasets_proton
    elif particle_type == 'muon':
        file_list = datasets_muon
    else:
        assert False, "particle_type can be either 'kaon' or 'pion'"
    
    if log:
        print("Reading and concatenating datasets:")
        for fname in file_list: print("\t{}".format(fname))
    data_full = load_and_merge_and_cut(file_list)
    # Must split the whole to preserve train/test split""
    if log: print("splitting to train/val/test")
    data_train, data_val, _ = split(data_full)
    if log: print("fitting the scaler")
    print("scaler train sample size: {}".format(len(data_train)))
    scaler = QuantileTransformer(output_distribution="normal",
                                 n_quantiles=int(1e5),
                                 subsample=int(1e10)).fit(data_train.drop(weight_col, axis=1).values)
    if log: print("scaling train set")
    data_train = pd.concat([scale_pandas(data_train.drop(weight_col, axis=1), scaler), data_train[weight_col]], axis=1)
    if log: print("scaling test set")
    data_val = pd.concat([scale_pandas(data_val.drop(weight_col, axis=1), scaler), data_val[weight_col]], axis=1)
    if dtype is not None:
        if log: print("converting dtype to {}".format(dtype))
        data_train = data_train.astype(dtype, copy=False)
        data_val = data_val.astype(dtype, copy=False)
    return data_train, data_val, scaler

#def get_scaled_dataset(dataset_index):
#    data_full = pd.read_hdf(datasets[dataset_index])
#    data_full = data_full[(data_full.particle_one_energy > ENERGY_CUT)]
#    p1_type_orig = np.copy(data_full.particle_one_type.values)
#    p2_type_orig = np.copy(data_full.particle_two_type.values)
#    # TOOD rewrite notebooks and use scaler on train/test
#    scaler = QuantileTransformer(output_distribution="normal",
#                                 n_quantiles=int(1e6),
#                                 subsample=int(1e10), copy=False)
#    data_full_np = scaler.fit_transform(data_full)
#    res = pd.DataFrame(data_full_np, columns=data_full.columns)
#    res.particle_one_type = p1_type_orig
#    res.particle_two_type = p2_type_orig
#    return res
