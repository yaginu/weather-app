from typing import NamedTuple

# 現行モデルが保存されているディレクトリを取得する
def get_default_model(
    project_id:str, model_id:str
) -> NamedTuple('Outputs', [('default_model_url', str)]):
    
    from googleapiclient import discovery
    from googleapiclient import errors

    service = discovery.build('ml', 'v1')
    name = 'projects/{}/models/{}'.format(project_id, model_id)

    response = service.projects().models().get(name=name).execute()
    
    default_model_uri = response["defaultVersion"]["deploymentUri"]
    
    return (default_model_uri)