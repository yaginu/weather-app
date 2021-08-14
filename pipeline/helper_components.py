from typing import NamedTuple

# ベストパフォーマンスのパラメータを取得する
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

# デフォルトモデルを変更する
def set_default_version(
    model_id: str, version_id: str
):
    from googleapiclient import discovery
    from googleapiclient import errors
    
    service = discovery.build('ml', 'v1')
    name = '{}/versions/{}'.format(model_id, version_id)
    
    print(name)

    response = service.projects().models().versions().setDefault(name=name).execute()