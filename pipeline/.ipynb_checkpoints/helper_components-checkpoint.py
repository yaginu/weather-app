from typing import NamedTuple

def retrieve_best_run(
    project_id:str, job_id:str
) -> NamedTuple('Outputs', [('metric_value', float), ('learning_rate', float), ('dropout_rate', float)]):
    
    from googleapiclient import discovery
    from googleapiclient import errors
    
    ml = discovery.build('ml', 'v1')
    
    job_name = 'projects/{}/jobs/{}'.format(project_id, job_id)
    request = ml.projects().jobs().get(name=job_name)
    
    try:
        response = request.execute()
        print(response)   
    except errors.HttpError as err:
        print(err)
    except:
        print('Unexpected error')    
    
    best_trial = response['trainingOutput']['trials'][0]
    
    print("best_trial:", best_trial)
    
    metric_value = best_trial['finalMetric']['objectiveValue']
    learning_rate = float(best_trial['hyperparameters']['learning_rate'])
    dropout_rate = float(best_trial['hyperparameters']['dropout_rate'])
    
    return (metric_value, learning_rate, dropout_rate)

# モデルの予測値と、実現値を使ってモデルの評価を行う
def evaluate_model(
    dataset_path: str, model_path: str, transform_artefacts_dir: str, metric_name: str
) -> NamedTuple('Outputs', [('metric_name', str), ('metric_value', float), ('mlpipeline_metrics', 'Metrics')]):
    
    import json
    
    import tensorflow as tf
    import numpy as np
    
    from create_dataset import load_test_dataset
    
    
    def calculate_loss(y_pred, y_true):
        
        mse = tf.keras.losses.MeanSquaredError()
        
        return mse(y_true, y_pred).numpy().astype(np.float64)
    
    model_path = '{}/predict'.format(model_path)
    model = tf.keras.models.load_model(model_path)
        
    x_test, y_true = load_test_dataset(dataset_path + "*", 256)
    
    x_test_transformed = x_test.map(model.preprocessing_layer)
    
    prediction = []
    for item in x_test_transformed:
        prediction.append(model.predict(item))
            
    y_pred = np.array(prediction).reshape(-1, 24)        
    y_true = np.array(list(tf.data.Dataset.as_numpy_iterator(y_true))).reshape(-1, 24)
        
    if metric_name == "mse":
        metric_value = calculate_loss(y_pred, y_true)
        print("metric_value:", metric_value)
                
    else:
        metric_name = 'N/A'
        metric_value = 0
        
    metrics = {
        'metrics': [{
            'name': metric_name,
            'numberValue': metric_value
        }]
    }
    
    return (metric_name, metric_value, json.dumps(metrics))
    