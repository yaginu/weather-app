import os
import pickle

import numpy as np
import tensorflow as tf

import preprocess


class MyPredictor(object):
    def __init__(self, model):
        self._model = model
        
    def predict(self, instances, **kwargs):
                
        preprocessed_inputs = {}
              
        for i in instances:
            for k, v in i.items():
                if k not in preprocessed_inputs.keys():
                    preprocessed_inputs[k] = [v]
                else:
                    preprocessed_inputs[k].append(v)
                            
        preprocessed_inputs["Date"] = [preprocess.convert_to_timestamp(i) for i in preprocessed_inputs["Date"]]
        preprocessed_inputs["wind_direction"] = [preprocess.direction_to_degree(i) for i in preprocessed_inputs["wind_direction"]]

        preprocessed_inputs = {
            k: tf.reshape(np.array(v, dtype=np.float32), shape=(-1, 24))
            for k, v in preprocessed_inputs.items()
        }
        
        transformed_inputs = self._model.preprocessing_layer(preprocessed_inputs)
        
        outputs = self._model.predict(transformed_inputs, steps=1).reshape(-1, 24)
                            
        return outputs.tolist()
    
    @classmethod
    def from_path(cls, model_dir):
        
        model = tf.keras.models.load_model(model_dir)    
         
        return cls(model)