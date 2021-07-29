from functools import partial

import tensorflow as tf


# Loading dataset
def load_dataset(filename, batch_size, mode):

    # Setting defaults
    CSV_COLUMNS = [
        'day_sin', 'day_cos', 'year_sin', 'year_cos', 'air_pressure_ashore', 'air_pressure_afloat', 'diff_air_pressure',
        'precipitation', 'temperature', 'humidity', 'wind_vector_x', "wind_vector_y", 
        'hours_of_daylight', 'global_solar_radiation', 'temp_mean', 'temp_var'
    ]

    SELECT_COLUMNS = [
        'day_sin', 'day_cos', 'year_sin', 'year_cos', 'air_pressure_ashore', 'air_pressure_afloat', 'diff_air_pressure',
        'precipitation', 'temperature', 'humidity', 'wind_vector_x', "wind_vector_y", 'hours_of_daylight', 'global_solar_radiation'
    ]

    DEFAULTS = [[0.0] for _ in range(len(SELECT_COLUMNS))]
    
    # Packing features
    def pack(features):
        packed_features =  tf.stack(list(features.values()), axis=1)

        return tf.reshape(packed_features, [-1])
    
    @tf.function
    def marshal(x, feature_keys):
        features = {
            k: x[:, feature_keys.index(k)] for k in feature_keys #pack時失われたkeyを付け直す

        }
        
        return features

    # Window processing
    def windowed_dataset(dataset, batch_size, mode):
        
        marshal_fn_partial = partial(marshal, feature_keys=SELECT_COLUMNS) 
        
        dataset = dataset.map(pack)
        dataset = dataset.window(size=48, shift=1, drop_remainder=True)
        dataset = dataset.flat_map(lambda window: window.batch(48))

        if mode == "train":
            dataset.shuffle(1000)
        
        encoder_input = dataset.map(lambda window: window[:24]).map(marshal_fn_partial)
        decoder_input = dataset.map(lambda window: tf.concat((tf.zeros((1)), window[24:-1, 8]), axis=0)) #Teacher Forcingのため、decoder_inputの先頭は、0にする
        decoder_output = dataset.map(lambda window: window[24:, 8])

        inputs = tf.data.Dataset.zip((encoder_input, decoder_input))
        dataset = tf.data.Dataset.zip((inputs, decoder_output)).cache()
            
        dataset = dataset.batch(batch_size, drop_remainder=True).repeat(1).prefetch(1)  
        
        return dataset
    
    dataset = tf.data.experimental.make_csv_dataset(
            file_pattern=filename,
            column_names=CSV_COLUMNS,
            column_defaults=DEFAULTS,
            select_columns=SELECT_COLUMNS,
            batch_size=1,
            shuffle=False,
            header=False,
            num_epochs=1)

    dataset = windowed_dataset(dataset, batch_size, mode)

    return dataset