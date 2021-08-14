from typing import NamedTuple

# モデルの予測値と、実現値を使ってモデルの評価を行う
def evaluate_model(
    project_id: str, model_id: str, dataset_path: str, model_path: str, metric_name: str
) -> NamedTuple('Outputs', [('metric_value', float), ('metric_value_default', float)]):
        
    import tensorflow as tf
    import numpy as np
    
    from create_dataset import load_test_dataset
    from get_default import get_default_model
    
    
    def calculate_loss(y_pred, y_true):
        
        mse = tf.keras.losses.MeanSquaredError()
        
        return mse(y_true, y_pred).numpy().astype(np.float64)
    
    # データセットの作成
    x_test, y_true = load_test_dataset(dataset_path + "*", 256)
    y_true = np.array(list(tf.data.Dataset.as_numpy_iterator(y_true))).reshape(-1, 24)

    # 訓練済みモデルでの予測
    model_path = '{}/predict'.format(model_path)
    model = tf.keras.models.load_model(model_path)

    x_test_transformed = x_test.map(model.preprocessing_layer)

    prediction = []
    for item in x_test_transformed:
        prediction.append(model.predict(item))
            
    y_pred = np.array(prediction).reshape(-1, 24)
    
    # 現行モデルでの予測
    try:
        default_model_path = get_default_model(project_id, model_id)
        default_model = tf.keras.models.load_model(default_model_path)

        x_test_transformed_default = x_test.map(default_model.preprocessing_layer)

        prediction_default = []
        for item in x_test_transformed_default:
            prediction_default.append(default_model.predict(item))

        y_pred_default = np.array(prediction_default).reshape(-1, 24)

    except:
        default_model_path = None
    
    if metric_name == "mse":
        
        metric_value = calculate_loss(y_pred, y_true)
        print("metric_value:", metric_value)
        
        # デプロイ済みモデルがある場合
        if default_model_path:
            metric_value_default = calculate_loss(y_pred_default, y_true)
            print("metric_value_default:", metric_value_default)

        # デプロイ済みモデルがない場合
        else:
            metric_value_default = float('inf')
                
    else:
        metric_name = 'N/A'
        metric_value = 0
        metric_value_default = 0
    
    return (metric_value, metric_value_default)
    