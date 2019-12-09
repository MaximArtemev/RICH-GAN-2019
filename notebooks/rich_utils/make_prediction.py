import os, sys
from collections import namedtuple
from typing import List, Optional, Dict

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.externals import joblib

# sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import rich_utils.utils_rich_mrartemev as utils_rich

tf_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

# Single object to store train, val and test data splits in
DataSplits = namedtuple("DataSplits", ['train', 'val', 'test'])


def load_and_predict(
        particles : List[str] = utils_rich.PARTICLES,
        models_path : str = 'exported_model',
        preprocessors_path : str = 'preprocessors',
        model_name_format : str = "FastFastRICH_Cramer_{}",
        preprocessor_name_format : str = "FastFastRICH_Cramer_{}_preprocessor.pkl",
        tf_config : tf.ConfigProto = tf_config,
        output_filename : Optional[str] = None,
        return_full : Optional[bool] = False
    ) -> Dict[str, DataSplits]:
    """
    Load data and models, make prediction and return augmented data
    """


    data_full = {
        particle : utils_rich.load_and_merge_and_cut(utils_rich.datasets[particle])
        for particle in particles
    }

    def _transform(arr, scaler, inverse=False):
        """Utility function to apply transfomations quickly"""
 
        t = scaler.inverse_transform if inverse else scaler.transform
        XY_t = t(arr[:,:-1])
        W = arr[:,-1:]
        return np.concatenate([XY_t, W], axis=1)
  
  
    ######### Loading models and making predictions #########
    for particle in particles:
        print(os.path.join(models_path, model_name_format.format(particle)))
        with tf.Session(config=tf_config) as sess:
            predictor = tf.contrib.predictor.from_saved_model(
                os.path.join(models_path, model_name_format.format(particle))
            )
            scaler = joblib.load(
                os.path.join(preprocessors_path, preprocessor_name_format.format(particle))
            )
            
            data_full_t = _transform(data_full[particle].values, scaler)
            
            batch_predictions = []
            for batch in np.array_split(data_full_t, len(data_full_t)//50000):
                batch_predictions.append(predictor({'x' : batch[:,utils_rich.y_count:]})['dlls'])

            data_full_t[:,:utils_rich.y_count] = np.row_stack(batch_predictions)
            data_full_predicted = _transform(data_full_t, scaler, inverse=True)
            
            for i, col in enumerate(utils_rich.dll_columns):
                data_full[particle]["predicted_{}".format(col)] = data_full_predicted[:,i]
                
    if return_full:
        return data_full

    
    splits = {
        particle : DataSplits(*utils_rich.split(data_full[particle]))
        for particle in particles
    }

    if not output_filename is None:
        print("Writing result to '{}'".format(output_filename))
        pd.to_pickle(splits, output_filename)
    return splits

def load_all_and_predict(
        models_path : str = 'exported_model',
        preprocessors_path : str = 'preprocessors',
        model_name_format : str = "FastFastRICH_Cramer_{}",
        preprocessor_name_format : str = "FastFastRICH_Cramer_{}_preprocessor.pkl",
        tf_config : tf.ConfigProto = tf_config,
        output_filename : Optional[str] = None
    ) -> Dict[str, DataSplits]:
    """
    Load data and models, make prediction and return augmented data
    """

    particles = utils_rich.PARTICLES
    data_full = {
        particle : utils_rich.load_and_merge_and_cut(utils_rich.datasets[particle])
        for particle in particles
    }    

    def _transform(arr, scaler, particle_ind, inverse=False):
        """Utility function to apply transfomations quickly"""
        W = arr[:,-1:]
        particle_ind_vec = np.full((len(arr), 1), particle_ind)
        XY_t = None
        if not inverse:
            XY_t = scaler.transform(arr[:,:-1])
        else:
            XY_t = scaler.inverse_transform(np.concatenate([arr[:, :utils_rich.y_count],
                                                            arr[:, utils_rich.y_count+1:-1]], axis=1))
        return np.concatenate([XY_t[:, :utils_rich.y_count],
                               particle_ind_vec,
                               XY_t[:, utils_rich.y_count:],
                               W], axis=1)
  
  
    ######### Loading models and making predictions #########
    with tf.Session(config=tf_config) as sess:
        predictor = tf.contrib.predictor.from_saved_model(
                    os.path.join(models_path, model_name_format)
        )
        for particle_ind, particle in enumerate(particles):
            print(os.path.join(models_path, model_name_format.format(particle)))
            with tf.Session(config=tf_config) as sess:
                scaler = joblib.load(
                    os.path.join(preprocessors_path, preprocessor_name_format.format(particle))
                )
                data_full_t = _transform(data_full[particle].values, scaler, particle_ind)
                ohe_table = np.zeros((len(data_full_t), len(particles)))
                ohe_table[:, particle_ind] = 1.
                     
                data_full_categorical = np.column_stack([data_full_t[:, :utils_rich.y_count],
                                         ohe_table,
                                         data_full_t[:, utils_rich.y_count:]])

                batch_predictions = []
                for batch in np.array_split(data_full_categorical, len(data_full_categorical)//50000):
                    batch_predictions.append(predictor({'x' : batch[:,utils_rich.y_count:-1]})['dlls'])

                data_full_t[:,:utils_rich.y_count] = np.row_stack(batch_predictions)
                data_full_predicted = _transform(data_full_t, scaler, particle_ind, inverse=True)

                for i, col in enumerate(utils_rich.dll_columns):
                    data_full[particle]["predicted_{}".format(col)] = data_full_predicted[:,i]

    splits = {
        particle : DataSplits(*utils_rich.split(data_full[particle]))
        for particle in particles
    }


    if not output_filename is None:
        print("Writing result to '{}'".format(output_filename))
        pd.to_pickle(splits, output_filename)
    return splits


