import tensorflow as tf
import tensorflow_addons as tfa

# Creating model for training and evaluating
def train_model(num_units=128, learning_rate=0.001, dropout_rate=0.35):
    
    SELECT_COLUMNS = [
        'day_sin', 'day_cos', 'year_sin', 'year_cos', 'air_pressure_ashore', 'air_pressure_afloat', 'diff_air_pressure',
        'precipitation', 'temperature', 'humidity', 'wind_vector_x', "wind_vector_y", 'hours_of_daylight', 'global_solar_radiation'
    ]
    
    # Input layer
    # tf.keras.experimental.SequenceFeaturesでの入力層は、モデルの保存ができず断念
    encoder_input_layers = {
        colname: tf.keras.layers.Input(name=colname, shape=(24, 1), dtype=tf.float32)
        for colname in SELECT_COLUMNS
    }
    
    pre_model_input = tf.keras.layers.Concatenate(axis=-1, name="concatenate")(encoder_input_layers.values())
    
    # Encoder
    encoder_lstm = tf.keras.layers.LSTM(num_units, return_sequences=True, name="encoder_lstm1")(pre_model_input)
    encoder_dropout = tf.keras.layers.Dropout(dropout_rate, name="encoder_dropout")(encoder_lstm)
    encoder_output, state_h, state_c = tf.keras.layers.LSTM(num_units, return_state=True, name="encoder_lstm2")(encoder_dropout)
    encoder_state = [state_h, state_c]

    # Scheduled Sampler
    sampler = tfa.seq2seq.sampler.ScheduledOutputTrainingSampler(
        sampling_probability=0.,
        next_inputs_fn=lambda outputs: tf.reshape(outputs, shape=(1, 1))
    )
    sampler.sampling_probability = tf.Variable(0.)

    # Decoder
    decoder_input = tf.keras.layers.Input(shape=(24, 1), name="decoder_input")

    decoder_cell = tf.keras.layers.LSTMCell(num_units, name="decoder_lstm")
    output_layer = tf.keras.layers.Dense(1, name="decoder_output")

    decoder = tfa.seq2seq.basic_decoder.BasicDecoder(decoder_cell, sampler, output_layer=output_layer)
    decoder_output, _, _ = decoder(decoder_input, initial_state=encoder_state, sequence_length=[24])

    final_output = decoder_output.rnn_output

    # Creating model
    model = tf.keras.Model(
        inputs=[encoder_input_layers, decoder_input], outputs=[final_output])

    optimizer = tf.keras.optimizers.RMSprop(learning_rate)
    model.compile(loss="mse", optimizer=optimizer)

    return model, encoder_input_layers, encoder_state, decoder_cell, output_layer, sampler


# Creating model for prediction
# Functional APIではモデルの保存時にエラーが出るので、サブクラス化する
def predict_model(encoder_input_layers, encoder_state, decoder_cell, output_layer):
    
    # Encoder Layer Class
    class Inference_Encoder(tf.keras.layers.Layer):
        def __init__(self, encoder_input_layers, encoder_state):
            super().__init__()

            self.model = tf.keras.models.Model(inputs=[encoder_input_layers], outputs=encoder_state)

        @tf.function
        def call(self, inputs):

            return self.model(inputs)

    # Decoder Layer Class
    class Inference_Decoder(tf.keras.layers.Layer):

        def __init__(self, decoder_cell, output_layer):
            super().__init__()

            # Inference sampler
            self.sampler = tfa.seq2seq.sampler.InferenceSampler(
                sample_fn = lambda outputs: tf.reshape(outputs, (1, 1)),
                sample_shape = [1],
                sample_dtype = tf.float32,
                end_fn = lambda sample_ids : False,
            )

            self.decoder = tfa.seq2seq.basic_decoder.BasicDecoder(
                decoder_cell, self.sampler, output_layer=output_layer, maximum_iterations=24
            )

        @tf.function
        def call(self, initial_state):
            start_inputs = tf.zeros(shape=(1, 1))
            decoder_output, _, _ = self.decoder(start_inputs, initial_state=initial_state)
            final_output = decoder_output.rnn_output

            return final_output

    # Inference Model Class
    class Inference_Model(tf.keras.Model):
        def __init__(self, encoder_input_layers, encoder_state, decoder_cell, output_layer):
            super().__init__()

            self.encoder = Inference_Encoder(encoder_input_layers, encoder_state)
            self.decoder = Inference_Decoder(decoder_cell, output_layer)

        @tf.function
        def call(self, inputs):

            inputs_copy = inputs.copy()
            
            # inputsは、transform_fnで処理したデータで、訓練セットの平均と分散が含まれている
            # rescaleのために、それらの統計量を取り出しておく
            temp_mean = inputs_copy.pop('temp_mean')[0][0]
            temp_var = inputs_copy.pop('temp_var')[0][0]

            initial_state = self.encoder(inputs_copy)
            outputs = self.decoder(initial_state)
            
            outputs_rescaled = outputs * tf.sqrt(temp_var) + temp_mean

            return outputs_rescaled
        
    inference_model = Inference_Model(encoder_input_layers, encoder_state, decoder_cell, output_layer)

    return inference_model