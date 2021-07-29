import tensorflow as tf

from functools import partial


# Loading dataset
def load_test_dataset(filename, batch_size):

    CSV_COLUMNS = [
        'Date', 'air_pressure_ashore', 'air_pressure_afloat', 'precipitation', 'temperature',
        'humidity', 'wind_direction', 'wind_velocity', 'hours_of_daylight', 'global_solar_radiation',
        'weather', 'cloud cover'
    ]

    SELECT_COLUMNS = [
        'Date', 'air_pressure_ashore', 'air_pressure_afloat', 'precipitation', 'temperature',
        'humidity', 'wind_direction', 'wind_velocity', 'hours_of_daylight', 'global_solar_radiation'
    ]

    DEFAULTS = [[0.0] for i in SELECT_COLUMNS]

    # Packing features
    def pack(features):
        packed_features =  tf.stack(list(features.values()), axis=1)

        return tf.reshape(packed_features, [-1])
    
    @tf.function
    def marshal(x, feature_keys):
        features = {
            k: x[:, feature_keys.index(k)] for k in feature_keys
        }
        
        return features

    # Window processing
    def windowed_dataset(dataset, batch_size):
        
        marshal_fn_partial = partial(marshal, feature_keys=SELECT_COLUMNS) 

        dataset = dataset.map(pack)
        dataset = dataset.window(size=48, shift=1, drop_remainder=True)
        dataset = dataset.flat_map(lambda window: window.batch(48))
            
        x_test = dataset.map(lambda window: window[:24]).map(marshal_fn_partial).batch(batch_size, drop_remainder=True).repeat(1).prefetch(1)  
        y_true = dataset.map(lambda window: window[24:, 4]).batch(batch_size, drop_remainder=True).repeat(1).prefetch(1)  
        
        return x_test, y_true
    
    dataset = tf.data.experimental.make_csv_dataset(
            file_pattern=filename,
            column_names=CSV_COLUMNS,
            column_defaults=DEFAULTS,
            select_columns=SELECT_COLUMNS,
            header=False,
            batch_size=1,
            shuffle=False,
            num_epochs=1
    )

    x_test, y_true = windowed_dataset(dataset, batch_size)

    return x_test, y_true