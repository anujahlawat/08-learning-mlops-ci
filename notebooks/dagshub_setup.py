mlflow.set_tracking_uri('https://dagshub.com/anujahlawat.ds/08-learning-mlops-ci.mlflow')

import dagshub
dagshub.init(repo_owner='anujahlawat.ds', repo_name='08-learning-mlops-ci', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)