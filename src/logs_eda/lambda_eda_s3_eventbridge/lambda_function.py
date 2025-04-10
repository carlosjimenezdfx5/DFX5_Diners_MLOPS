'''
    Lambda Function to Logs EDA
    Release Date : 2025-04-09
'''
#######################
# ---- libraries ---- #
#######################
import json
###########################
# ---- Main Function ---- #
###########################

def lambda_handler(event, context):
    print("Evento recibido desde S3/EventBridge:")
    print(json.dumps(event, indent=2))
    return {
        'statusCode': 200,
        'body': json.dumps('Evento procesado correctamente')
    }