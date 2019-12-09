import os
import tensorflow as tf
from quantile_transformer_tf import QuantileTransformerTF
from rich_utils import utils_rich_mrartemev as utils_rich
import pandas as pd
import numpy as np
import time
from sklearn.externals import joblib
from rich_utils.make_prediction import DataSplits

particles = ['kaon', 'pion', 'proton', 'muon']
tf_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))


def attach_tf_scaler(
        particles = None,
        models_path = 'exported_model',
        export_models_path = 'exported_model',
        preprocessors_path = 'preprocessors',
        model_name_format = "FastFastRICH_Cramer_{}",
        preprocessor_name_format = "FastFastRICH_Cramer_{}_preprocessor.pkl",
        tf_config=tf_config,
        cuda='0'):
    if particles is None:
        particles = ['kaon', 'pion', 'proton', 'muon']
    
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda
    
    data_full = {
        particle : utils_rich.load_and_merge_and_cut(utils_rich.datasets[particle])
        for particle in particles
    }    

    for particle in particles:
        print("Working on {}s".format(particle))
        predictor = tf.contrib.predictor.from_saved_model(
                        os.path.join(models_path, model_name_format.format(particle)),
                        config=tf_config
                    )
        print("Loaded predictor")

        scaler = joblib.load(
            os.path.join(preprocessors_path, preprocessor_name_format.format(particle))
        )
        print("Loaded scaler")

        input_name, = [x.name for x in predictor.feed_tensors['x'].consumers()[0].outputs]
        output_name = predictor.fetch_tensors['dlls'].name

        mod_graph = tf.Graph()
        with mod_graph.as_default():
            tf_scaler_x = QuantileTransformerTF(scaler,
                                                list(range(utils_rich.y_count, scaler.quantiles_.shape[1])),
                                                np.float64)
            tf_scaler_y = QuantileTransformerTF(scaler,
                                                list(range(0, utils_rich.y_count)),
                                                np.float64)

            print("Created tf scalers")

            input_tensor = tf.placeholder(dtype=tf.float64, shape=(None, len(utils_rich.raw_feature_columns)), name='x')
            scaled_input = tf.cast(tf_scaler_x.transform(input_tensor, False), dtype=tf.float32)

            with tf.Session(config=tf_config) as sess:
                meta_graph_def = tf.saved_model.loader.load(
                    sess, ['serve'],
                    os.path.join(models_path, model_name_format.format(particle)),
                    input_map={input_name : scaled_input}
                )
                print("Reloaded the model with input_map")

                scaled_output = mod_graph.get_tensor_by_name(output_name)
                output_tensor = tf_scaler_y.transform(tf.cast(scaled_output, dtype=tf.float64), True)

                sub_graph_def = tf.graph_util.convert_variables_to_constants(
                    sess, sess.graph_def, [output_tensor.op.name]
                )

                # Removing unnecessary copies of data:
                rm_idx = []
                for i, func in enumerate(sub_graph_def.library.function):
                    if 'make_dataset' in func.signature.name:
                        rm_idx.append(i)
                rm_idx = sorted(rm_idx, reverse=True)
                for i in rm_idx:
                    sub_graph_def.library.function.remove(sub_graph_def.library.function[i])

                print("Sub-graph created. Size : O({:.3}) MB".format(len(str(sub_graph_def)) / 1024**2))

        reduced_graph = tf.Graph()
        with reduced_graph.as_default():
            with tf.Session(config=tf_config) as sess:
                tf.import_graph_def(sub_graph_def, name='')
                input_tensor  = reduced_graph.get_tensor_by_name(input_tensor.name)
                output_tensor = reduced_graph.get_tensor_by_name(output_tensor.name)
                print('Particle: {}, start time: {}'.format(particle, time.time()))
                predictions = sess.run(
                    output_tensor,
                    feed_dict={
                        input_tensor : data_full[particle][utils_rich.raw_feature_columns].values
                    }
                )
                print('end time: {}'.format(time.time()))
                print("Calculated predictions with the sub-graph")

                for i, col in enumerate(utils_rich.dll_columns):
                    data_full[particle]["predicted_{}".format(col)] = predictions[:,i]

                model_export_dir = os.path.join(export_models_path, model_name_format.format(particle) + "_tfScaler")
                tf.saved_model.simple_save(
                    sess, model_export_dir,
                    inputs={"x": input_tensor},
                    outputs={"dlls": output_tensor}
                )
                print("Exported the sub-graph model")

    splits = {
        particle : DataSplits(*utils_rich.split(data_full[particle]))
        for particle in particles
    }

    pd.to_pickle(splits, model_name_format.format(particle) + "_tfScaler.pkl")
    print("Saved predictions to pickle")
    return splits

def attach_tf_scaler_big_model(
        models_path = 'exported_model',
        export_models_path = 'exported_model',
        preprocessors_path = 'preprocessors',
        model_name_format = "FastFastRICH_Cramer_all_particles",
        preprocessor_name_format = "FastFastRICH_Cramer_all_particles/{}_preprocessor.pkl",
        tf_config=tf_config,
        cuda='0'):
    particles = ['kaon', 'pion', 'proton', 'muon']
    
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda
    
    data_full = {
        particle : utils_rich.load_and_merge_and_cut(utils_rich.datasets[particle])
        for particle in particles
    }
    data_full_indexed = {
        particle: pd.concat([data_full[particle].iloc[:, :utils_rich.y_count],
                            pd.DataFrame({'particle': np.full((len(data_full[particle])), index)}),
                            data_full[particle].iloc[:, utils_rich.y_count:]], axis=1)
        for index, particle in enumerate(particles)
    }
    data_full_indexed_stacked = np.concatenate([data_full_indexed[particle] for particle in particles], axis=0) 
                            
    
    predictor = tf.contrib.predictor.from_saved_model(
                os.path.join(models_path, model_name_format),
                config=tf_config
            )
    print("Loaded predictor")
    
    sklearn_scalers = []
    
    for particle_ind, particle in enumerate(particles):
        sklearn_scalers.append(joblib.load(
            os.path.join(preprocessors_path, preprocessor_name_format.format(particle))
        ))
        print("Loaded {}s scaler".format(particle))

    input_name, = [x.name for x in predictor.feed_tensors['x'].consumers()[0].outputs]
    output_name = predictor.fetch_tensors['dlls'].name

    mod_graph = tf.Graph()
        
    # Сюда хочется добавить вектор с номерами частицы
    with mod_graph.as_default():
        total_data = []
        input_tensor = tf.placeholder(dtype=tf.float64, shape=(None, len(utils_rich.raw_feature_columns) + 1), name='x')

        for particle_ind, particle in enumerate(particles):
            scaler = joblib.load(
                os.path.join(preprocessors_path, preprocessor_name_format.format(particle))
            )
            print("Loaded {}s scaler".format(particle))

            tf_scaler_x = (QuantileTransformerTF(scaler,
                                            list(range(utils_rich.y_count, scaler.quantiles_.shape[1])),
                                            np.float64))
            tf_scaler_y = (QuantileTransformerTF(scaler,
                                            list(range(0, utils_rich.y_count)),
                                            np.float64))
            
            index_mask = tf.math.equal(data_full_indexed_stacked[:, -1], particle_ind)
            transformed_data = tf.cast(tf_scaler_x.transform(tf.boolean_mask(data_full_indexed_stacked[:, :-1],
                                                                                            index_mask), False), dtype=tf.float32)
            total_data.append(transformed_data)
        scaled_input = tf.cast(tf.concat(total_data, axis=0), dtype=tf.float32)
        
        print("Created tf scalers")
        
        with tf.Session(config=tf_config) as sess:
            meta_graph_def = tf.saved_model.loader.load(
                sess, ['serve'],
                os.path.join(models_path, model_name_format),
                input_map={input_name : scaled_input}
            )
            print("Reloaded the model with input_map")

            scaled_output = mod_graph.get_tensor_by_name(output_name)
            output_tensor = tf_scaler_y.transform(tf.cast(scaled_output, dtype=tf.float64), True)

            sub_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, sess.graph_def, [output_tensor.op.name]
            )

            # Removing unnecessary copies of data:
            rm_idx = []
            for i, func in enumerate(sub_graph_def.library.function):
                if 'make_dataset' in func.signature.name:
                    rm_idx.append(i)
            rm_idx = sorted(rm_idx, reverse=True)
            for i in rm_idx:
                sub_graph_def.library.function.remove(sub_graph_def.library.function[i])

            print("Sub-graph created. Size : O({:.3}) MB".format(len(str(sub_graph_def)) / 1024**2))

    reduced_graph = tf.Graph()
    with reduced_graph.as_default():
        with tf.Session(config=tf_config) as sess:
            tf.import_graph_def(sub_graph_def, name='')
            input_tensor  = reduced_graph.get_tensor_by_name(input_tensor.name)
            output_tensor = reduced_graph.get_tensor_by_name(output_tensor.name)
            print('Particle: {}, start time: {}'.format(particle, time.time()))
            predictions = sess.run(
                output_tensor,
                feed_dict={
                    input_tensor : data_full[particle][utils_rich.raw_feature_columns].values # here
                }
            )
            print('end time: {}'.format(time.time()))
            print("Calculated predictions with the sub-graph")

            for i, col in enumerate(utils_rich.dll_columns):
                data_full[particle]["predicted_{}".format(col)] = predictions[:,i]

            model_export_dir = os.path.join(export_models_path, model_name_format.format(particle) + "_tfScaler")
            try:
                tf.saved_model.simple_save(
                    sess, model_export_dir,
                    inputs={"x": input_tensor},
                    outputs={"dlls": output_tensor}
                )
                print("Exported the sub-graph model")
            except Exception as e:
                print(e)

    splits = {
        particle : DataSplits(*utils_rich.split(data_full[particle]))
        for particle in particles
    }

    pd.to_pickle(splits, model_name_format.format(particle) + "_tfScaler.pkl")
    print("Saved predictions to pickle")
    return splits