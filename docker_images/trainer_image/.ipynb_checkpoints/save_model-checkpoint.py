import tensorflow as tf

def export_serving_model(model, tf_transform_output, out_dir):

    TRANSFORM_FEATURE_COLUMNS = [
        'Date', 'air_pressure_ashore', 'air_pressure_afloat', 'precipitation', 'temperature',
        'humidity', 'wind_direction', 'wind_velocity', 'hours_of_daylight', 'global_solar_radiation'
    ]

    SELECT_COLUMNS = [
        'day_sin', 'day_cos', 'year_sin', 'year_cos', 'air_pressure_ashore', 'air_pressure_afloat',
        'diff_air_pressure', 'precipitation', 'temperature', 'humidity', 'wind_vector_x', "wind_vector_y",
        'hours_of_daylight', 'global_solar_radiation', 'temp_mean', 'temp_var'
    ]
    
    # Building Model
    example = {
        x: tf.random.uniform(shape=(1, 24), name=x)
        for x in SELECT_COLUMNS
    }
    ex = model(example)
    
    # Transform raw features
    def get_apply_tft_layer(tf_transform_output):
        
        tft_layer = tf_transform_output.transform_features_layer()

        @tf.function
        def apply_tf_transform(raw_features_dict):

            unbatched_raw_features = {
                k: tf.squeeze(tf.reshape(v, (1, -1)))
                for k, v in raw_features_dict.items()
            }

            transformed_dataset = tft_layer(unbatched_raw_features)

            expanded_dims = {
                k: tf.reshape(v, (-1, 24))
                for k, v in transformed_dataset.items()
            }
            
            return expanded_dims

        return apply_tf_transform

    def get_serve_raw_fn(model, tf_transform_output):

        model.preprocessing_layer = get_apply_tft_layer(tf_transform_output)

        @tf.function
        def serve_raw_fn(features):

            preprocessed_features = model.preprocessing_layer(features)
        
            return preprocessed_features

        return serve_raw_fn
    
    serving_raw_entry = get_serve_raw_fn(model, tf_transform_output)   
    
    serving_transform_signature_tensorspecs = {
        x: tf.TensorSpec(shape=[None, 24], dtype=tf.float32, name=x)
        for x in TRANSFORM_FEATURE_COLUMNS
    }

    serving_signature_tensorspecs = {
        x: tf.TensorSpec(shape=[None, 24], dtype=tf.float32, name=x)
        for x in SELECT_COLUMNS
    }
    
    # Signatures
    signatures = {'serving_default': model.call.get_concrete_function(serving_signature_tensorspecs),
                  'transform': serving_raw_entry.get_concrete_function(serving_transform_signature_tensorspecs)}

    tf.keras.models.save_model(model=model, filepath=out_dir, signatures=signatures)