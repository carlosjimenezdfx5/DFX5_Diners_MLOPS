"""
    Lambda Function to Logs EDA via Kafka
    Release Date : 2025-04-10
"""

import json
import boto3
import tempfile
import base64
from eda_logs import read_and_clean_logs, run_eda

def lambda_handler(event, context):
    """
    Función Lambda que procesa un evento Kafka con información del S3 path,
    ejecuta un análisis EDA sobre el CSV y guarda un HTML en el bucket.
    """
    for record in event['records']['logs-eda-topic']:
        payload = json.loads(base64.b64decode(record['value']).decode('utf-8'))

        bucket = payload['bucket']
        key = payload['key']

        s3 = boto3.client('s3')
        with tempfile.NamedTemporaryFile() as tmp:
            s3.download_file(bucket, key, tmp.name)
            _, df_res = read_and_clean_logs(tmp.name)
            output_html = run_eda(df_res)
            s3.upload_file(output_html, bucket, "EDA_results/eda_logs_output.html")

    return {
        'statusCode': 200,
        'body': json.dumps('EDA ejecutado correctamente desde Kafka')
    }