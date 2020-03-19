import logging
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler

logger = logging.getLogger('main.rich_utls.data_preprocessing')

data_dir = 'data/data_calibsample/'
if 'research.utils_rich_mrartemev' != __name__:
    data_dir = '../{}'.format(data_dir)

logger.debug(f"Data directory: {data_dir}")


def get_particle_dset(particle):
    return [data_dir + name for name in os.listdir(data_dir) if particle in name]


list_particles = ['kaon', 'pion', 'proton', 'muon', 'electron']
PARTICLES = list_particles
datasets = {particle: get_particle_dset(particle) for particle in list_particles}

# dll_collumns - ~logprobs of the particle.
dll_columns = ['RichDLLe', 'RichDLLk', 'RichDLLmu', 'RichDLLp', 'RichDLLbt']
logger.info(f"dll_collumns: {dll_columns}")
# raw_feature_columns - input features.
raw_feature_columns = ['Brunel_P', 'Brunel_ETA', 'nTracks_Brunel']
logger.info(f"raw_feature_columns: {raw_feature_columns}")
# weight_col - due to mixed signal/background the sPlot technique is used to compute event weights
weight_col = ['probe_sWeight']
logger.info(f"weight_col: {weight_col}")

y_count = len(dll_columns)
TEST_SIZE = 0.3


def load_and_cut(file_name):
    data = pd.read_csv(file_name, delimiter='\t')
    return data[dll_columns + raw_feature_columns + weight_col]


def load_and_merge_and_cut(filename_list):
    return pd.concat([load_and_cut(fname) for fname in filename_list], axis=0, ignore_index=True)


def split(data):
    # train - val split
    data_train, data_val = train_test_split(data, test_size=TEST_SIZE, random_state=42)
    return data_train.reset_index(drop=True), \
           data_val.reset_index(drop=True)


def scale_pandas(dataframe, scaler):
    return pd.DataFrame(scaler.transform(dataframe.values), columns=dataframe.columns)


def get_merged_typed_dataset(particle_type, dtype=None, n_quantiles=100000):
    file_list = datasets[particle_type]
    logger.info(f"Data files for {particle_type}: {file_list}")
    data_full = load_and_merge_and_cut(file_list)
    # Must split the whole to preserve train/test split""
    data_train, data_val = split(data_full)
    logger.debug(f"scaler train sample size: {len(data_train)}")
    if n_quantiles == 0:
        scaler = StandardScaler().fit(data_train.drop(weight_col, axis=1).values)
    else:
        scaler = QuantileTransformer(output_distribution="normal",
                                     n_quantiles=n_quantiles,
                                     subsample=int(1e10)).fit(data_train.drop(weight_col, axis=1).values)
    logger.debug(f"scaler n_quantiles: {n_quantiles}")
    data_train = pd.concat([scale_pandas(data_train.drop(weight_col, axis=1), scaler), data_train[weight_col]], axis=1)
    data_val = pd.concat([scale_pandas(data_val.drop(weight_col, axis=1), scaler), data_val[weight_col]], axis=1)
    if dtype is not None:
        logger.debug(f"converting dtype to {dtype}")
        data_train = data_train.astype(dtype, copy=False)
        data_val = data_val.astype(dtype, copy=False)
    return data_train, data_val, scaler
