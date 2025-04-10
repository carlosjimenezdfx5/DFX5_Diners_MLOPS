"""
    EDA Logs
    Release Date : 2025-04-10
"""

import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
import json

def read_and_clean_logs(csv_path):
    """
    Lee el archivo CSV con problemas de delimitación y comillas,
    y separa en filas de tipo Request y Response.
    Devuelve dos DataFrames: df_requests, df_responses.
    """
    lines = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        for line in f:
            cleaned_line = line.rstrip().rstrip(';').strip()
            if cleaned_line.startswith('"') and cleaned_line.endswith('"'):
                cleaned_line = cleaned_line[1:-1]
            cleaned_line = cleaned_line.replace('""','"')
            lines.append(cleaned_line)
    parsed_data = list(csv.reader(lines, delimiter=','))
    header = parsed_data[0]
    data_rows = parsed_data[1:]
    df_res_rows = []
    df_req_rows = []
    for row in data_rows:
        if len(row) == len(header):
            df_res_rows.append(row)
        else:
            df_req_rows.append(row)
    df_res = pd.DataFrame(df_res_rows, columns=header)
    df_req = pd.DataFrame(df_req_rows)
    return df_req, df_res

def run_eda(df_res):
    df_res['ambiente'] = df_res['ambiente'].replace('', np.nan)
    df_res['appname'] = df_res['appname'].replace('', np.nan)
    df_res['timestamp'] = pd.to_datetime(df_res['timestamp'], unit='s', errors='coerce')
    df_res['file'] = df_res['file'].replace('', np.nan)
    df_res['clase'] = df_res['clase'].replace('', np.nan)
    df_res['class'] = df_res['class'].replace('', np.nan)
    df_res['dns_id'] = df_res['dns_id'].replace('', np.nan)
    df_res['endPoint'] = df_res['endPoint'].replace('', np.nan)
    df_res['id'] = df_res['id'].replace('', np.nan)
    df_res['host'] = df_res['host'].replace('', np.nan)
    df_res['kubernetes'] = df_res['kubernetes'].replace('', np.nan)
    df_res['level'] = df_res['level'].str.capitalize()
    df_res['log'] = df_res['log'].replace('', np.nan)
    df_res['logger'] = df_res['logger'].replace('', np.nan)
    df_res['method'] = df_res['method'].replace('', np.nan)
    df_res['metodo'] = df_res['metodo'].replace('', np.nan)
    df_res['metodoHttp'] = df_res['metodoHttp'].replace('', np.nan)
    df_res['metodo_final'] = df_res['metodoHttp'].combine_first(df_res['metodo']).combine_first(df_res['method'])
    df_res['msg'] = df_res['msg'].replace('', np.nan)
    df_res['name'] = df_res['name'].replace('', np.nan)
    df_res['namespace'] = df_res['namespace'].replace('', np.nan)
    df_res['operacionId'] = df_res['operacionId'].replace('', np.nan)
    df_res['pod'] = df_res['pod'].replace('', np.nan)
    df_res['thread'] = df_res['thread'].replace('', np.nan)
    df_res['tipoLog'] = df_res['tipoLog'].replace('', np.nan).str.capitalize()
    df_res['tipoMensaje'] = df_res['tipoMensaje'].replace('', np.nan)
    df_res['transaccionId'] = df_res['transaccionId'].replace('', np.nan)
    df_res['trazaExcepcion'] = df_res['trazaExcepcion'].replace('', np.nan)
    df_res['ts'] = df_res['ts'].replace('', np.nan)
    df_res['versionAplicacion'] = df_res['versionAplicacion'].replace('', np.nan)
    df_res['codigoEstadoHttp'] = pd.to_numeric(df_res['codigoEstadoHttp'], errors='coerce')
    df_res['tiempoProceso'] = df_res['tiempoProceso'].str.replace('ms','', regex=False).str.strip()
    df_res['tiempoProceso'] = pd.to_numeric(df_res['tiempoProceso'], errors='coerce')
    df_res['timestamp_iso'] = pd.to_datetime(df_res['timestamp_iso'], errors='coerce')
    profile = ProfileReport(df_res[[
        'timestamp', 'appname', 'endPoint', 'metodo_final', 'codigoEstadoHttp',
        'tipoMensaje', 'tiempoProceso']].copy(),
        title="Reporte EDA Logs Banco", explorative=True)
    profile.to_file("eda_logs_output.html")
    return "eda_logs_output.html"

