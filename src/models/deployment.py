from azureml.core import Workspace


def deploy(ws, model_name):
    model = ws.models[model_name]
