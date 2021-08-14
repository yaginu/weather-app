import datetime
import os

import kfp
from kfp.components import func_to_container_op
from kfp.gcp import use_gcp_secret

from helper_components import retrieve_best_run
from helper_components import set_default_version
from compare_models import evaluate_model
from preprocess_dataflow_pipeline import run_transformation_pipeline

# Setting defaults
BASE_IMAGE = os.getenv('BASE_IMAGE')
TRANSFORM_IMAGE = os.getenv('TRANSFORM_IMAGE')
TRAINER_IMAGE = os.getenv('TRAINER_IMAGE')
EVALUATE_IMAGE = os.getenv('EVALUATE_IMAGE')
RUNTIME_VERSION = os.getenv('RUNTIME_VERSION')
PYTHON_VERSION = os.getenv('PYTHON_VERSION')
COMPONENT_URL_SEARCH_PREFIX = os.getenv('COMPONENT_URL_SEARCH_PREFIX')

HYPERTUNE_SETTINGS = """
{
    "hyperparameters": {
        "goal": "MINIMIZE",
        "maxTrials": 3,
        "maxParallelTrials": 3,
        "hyperparameterMetricTag": "val_loss",
        "enableTrialEarlyStopping": True,
        "params":[
            {
                "parameterName": "learning_rate",
                "type": "DOUBLE",
                "minValue": 0.00001,
                "maxValue": 0.1,
                "scaleType": "UNIT_LOG_SCALE"
            },
            {
                "parameterName": "dropout_rate",
                "type": "DOUBLE",
                "minValue": 0.1,
                "maxValue": 0.4,
                "scaleType": "UNIT_LOG_SCALE"
            }
        ]
    }
}
"""

        
# Create component factories        
component_store = kfp.components.ComponentStore(
    local_search_paths=None, url_search_prefixes=[COMPONENT_URL_SEARCH_PREFIX])

# Pre-build components
mlengine_train_op = component_store.load_component('ml_engine/train')
mlengine_deploy_op = component_store.load_component('ml_engine/deploy')

# Lightweight components
run_transform_pipeline_op = func_to_container_op(
    run_transformation_pipeline, base_image=TRANSFORM_IMAGE)
retrieve_best_run_op = func_to_container_op(
    retrieve_best_run, base_image=BASE_IMAGE)
evaluate_model_op = func_to_container_op(
    evaluate_model, base_image=EVALUATE_IMAGE)
set_default_model_op = func_to_container_op(
    set_default_version, base_image=BASE_IMAGE)

        
# Defining the pipeline
@kfp.dsl.pipeline(
    name='Weather-forecast Model Training',
    description='The pipeline training and deploying the Weather-forecast pipeline'
)
def weather_forecast_train(
        project_id,
        gcs_root,
        region,
        source_table_name,
        num_epochs_hypertune,
        num_epochs_retrain,
        num_units,
        evaluation_metric_name,
        evaluation_metric_threshold,
        model_id,
        version_id,
        replace_existing_version,
        hypertune_settings=HYPERTUNE_SETTINGS):

    # Creating datasets    
    job_name = 'preprocess-weather-features' + '-' + datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    dataset_location = '{}/{}/{}'.format(gcs_root, 'datasets', kfp.dsl.RUN_ID_PLACEHOLDER)

    create_dataset = run_transform_pipeline_op(
        source_table_name, job_name, gcs_root, project_id, region, dataset_location)

    # Tune hyperparameters
    tune_args = [
        '--training_dataset_path', create_dataset.outputs["training_file_path"],
        '--validation_dataset_path', create_dataset.outputs["validation_file_path"],
        '--num_epochs', num_epochs_hypertune,
        '--num_units', num_units,
        '--hptune', 'True',
        '--transform_artefacts_dir', create_dataset.outputs["transform_artefacts_dir"]
    ]
        
    job_dir = '{}/{}/{}'.format(gcs_root, 'jobdir/hypertune', kfp.dsl.RUN_ID_PLACEHOLDER)
        
    hypertune = mlengine_train_op(
        project_id=project_id,
        region=region,
        master_image_uri=TRAINER_IMAGE,
        job_dir=job_dir,
        args=tune_args,
        training_input=hypertune_settings).apply(use_gcp_secret()) # Kubernetesシークレットを使用しないと、長時間の訓練が途中で停止します。
    
    #Retrive the best trial
    get_best_trial = retrieve_best_run_op(
        project_id, hypertune.outputs['job_id'])

    # Re-training the model
    job_dir = '{}/{}/{}'.format(gcs_root, 'jobdir', kfp.dsl.RUN_ID_PLACEHOLDER)
                                
    train_args = [
        '--training_dataset_path', create_dataset.outputs["training_file_path"],
        '--validation_dataset_path', create_dataset.outputs["validation_file_path"],
        '--num_epochs', num_epochs_retrain,
        '--num_units', num_units,
        '--learning_rate', get_best_trial.outputs['learning_rate'],
        '--dropout_rate', get_best_trial.outputs['dropout_rate'],
        '--hptune', 'False',
        '--transform_artefacts_dir', create_dataset.outputs["transform_artefacts_dir"]
    ]
    
    train_model = mlengine_train_op(
        project_id=project_id,
        region=region,
        master_image_uri=TRAINER_IMAGE,
        job_dir=job_dir,
        args=train_args).apply(use_gcp_secret())
    

    # Evaluating the model
    eval_model = evaluate_model_op(
        project_id=project_id,
        model_id=model_id,
        dataset_path=create_dataset.outputs["testing_file_path"],
        model_path=str(train_model.outputs["job_dir"]),
        metric_name=evaluation_metric_name)
    
    # Deploying the model
    with kfp.dsl.Condition(
        eval_model.outputs['metric_value'] < eval_model.outputs['metric_value_default']
            and eval_model.outputs['metric_value'] < evaluation_metric_threshold
    ):

        model_uri = '{}/predict'.format(train_model.outputs["job_dir"])

        deploy_model = mlengine_deploy_op(
            model_uri=model_uri,
            project_id=project_id,
            model_id=model_id,
            version_id=version_id,
            model = {"regions": [region],
    #                  "onlinePredictionLogging": True, # 同名のモデルがあると、デプロイ時にエラーが出るので、コメントアウトします。
                     "onlinePredictionConsoleLogging": True},
            version = {"packageUris": ["gs://intrepid-hour-320405-kubeflowpipelines-default/staging/dist/my_custom_code-0.1.tar.gz"],
                       "predictionClass": "predictor.MyPredictor"},
            runtime_version=RUNTIME_VERSION,
            python_version=PYTHON_VERSION,
            replace_existing_version=replace_existing_version)

        # Changing the default model
        change_default = set_default_model_op(
            model_id=deploy_model.outputs["model_name"],
            version_id=version_id
        ).set_caching_options(enable_caching=False) # キャッシュを使うと上手くデフォルトが切り替わらないことがあります。