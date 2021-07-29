import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
import tensorflow_transform as tft
import fire
import hypertune

from create_dataset import load_dataset
from create_model import train_model
from create_model import predict_model
from save_model import export_serving_model

# Training and evaluating the model
def train_evaluate(job_dir, training_dataset_path, validation_dataset_path, num_epochs, num_units, learning_rate, dropout_rate, hptune, transform_artefacts_dir):
    
    training_dataset = load_dataset(training_dataset_path + "*", 256, "train")
    validation_dataset = load_dataset(validation_dataset_path + "*", 128, "eval")
    
    print('Starting training: learning_rate={}, dropout_rate={}'.format(learning_rate, dropout_rate))
    
    tf_transform_output = tft.TFTransformOutput(transform_artefacts_dir)
    
    model, encoder_input_layers, encoder_state, decoder_cell, output_layer, sampler = train_model(
        num_units=num_units, learning_rate=learning_rate, dropout_rate=dropout_rate
    )
    
    def update_sampling_probability(epoch, logs):
        eps = 1e-16
        proba = max(0.0, min(1.0, epoch / (num_epochs - 10 + eps)))
        sampler.sampling_probability.assign(proba)

    sampling_probability_cb = tf.keras.callbacks.LambdaCallback(on_epoch_begin=update_sampling_probability)
    
    history = model.fit(training_dataset,
            epochs=num_epochs,
            validation_data=validation_dataset,
            callbacks=[sampling_probability_cb]
            )
    
    # Hyperparameter tuning
    if hptune:
        val_loss = history.history["val_loss"]
        print("val_loss: {}".format(val_loss))
        
        hpt = hypertune.HyperTune()
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag='val_loss',
            metric_value=val_loss[-1])
     
    # Saving the model
    if not hptune:
        inference_model_dir = '{}/predict'.format(job_dir)
        inference_model = predict_model(encoder_input_layers, encoder_state, decoder_cell, output_layer)
        export_serving_model(inference_model, tf_transform_output, inference_model_dir)
    
        print('Inference model saved in: {}'.format(inference_model_dir))

# Execution 
if __name__ == '__main__':    
    fire.Fire(train_evaluate)