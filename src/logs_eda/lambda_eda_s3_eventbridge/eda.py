'''
    EDA Logs
    Release Date : 2025-04-10
'''

#=====================#
# ---- Libraries ---- #
#=====================#

import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# 1. LECTURA Y LIMPIEZA DE LA DATA
# ---------------------------------------------------------------------

def read_and_clean_logs(csv_path):
    """
    Lee el archivo CSV 'muestra_logs.csv' con problemas de delimitación y comillas,
    y separa automáticamente en filas de tipo Request y Response.
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

    # Observamos que por el formato sucio, las filas de "Request" y "Response"
    # tienen distinta longitud. Ejemplo:
    #  - Response: 39 columnas
    #  - Request:  47 columnas
    # Separamos con base en la longitud que concuerde con la del header "oficial".
    # (Podríamos haber creado un header aparte para requests, pero se simplifica aquí).

    df_res_rows = []
    df_req_rows = []
    for row in data_rows:
        # Chequeamos la primera o la segunda fila; en pruebas se notó
        # que "Response" venía con 39 col. y "Request" con 47 col.
        if len(row) == len(header):
            df_res_rows.append(row)
        else:
            df_req_rows.append(row)

    # Construimos DataFrame de responses
    df_res = pd.DataFrame(df_res_rows, columns=header)

    # Para los requests no tenemos un "header" claro que coincida,
    # así que se ejemplifica con un DataFrame "sin nombre de columnas"
    # con la idea de limpiarlo manualmente. Queda a tu preferencia:
    df_req = pd.DataFrame(df_req_rows)

    return df_req, df_res


# Llama a la función de lectura/limpieza
csv_path = 'muestra_logs.csv'  # Ajusta la ruta si hace falta
df_req, df_res = read_and_clean_logs(csv_path)

print("Cantidad de registros (posibles requests):", len(df_req))
print("Cantidad de registros (responses):", len(df_res))
print("\nEjemplo df_res (respuestas):")
print(df_res.head())

# ---------------------------------------------------------------------
# 2. LIMPIEZA ADICIONAL Y CONVERSIÓN DE COLUMNAS (df_res)
# ---------------------------------------------------------------------

# Veamos qué columnas tiene df_res
print("\nColumnas df_res:", df_res.columns.tolist())

# Convirtamos a numérico las que nos interesan
df_res['codigoEstadoHttp'] = pd.to_numeric(df_res['codigoEstadoHttp'], errors='coerce')

# 'tiempoProceso' puede venir con "ms" al final, lo limpiamos.
df_res['tiempoProceso'] = (
    df_res['tiempoProceso']
    .str.replace('ms','', regex=False)
    .str.strip()
)
df_res['tiempoProceso'] = pd.to_numeric(df_res['tiempoProceso'], errors='coerce')

# También podríamos querer parsear el "timestamp_iso" como fecha real
df_res['timestamp_iso'] = pd.to_datetime(df_res['timestamp_iso'], errors='coerce')

# Verifiquemos
print("\nTipos de dato df_res:")
print(df_res.dtypes)

# ---------------------------------------------------------------------
# 3. EXPLORACIÓN INICIAL (EDA) - DESCRIPTIVOS
# ---------------------------------------------------------------------

print("\n--- Estadísticas descriptivas de df_res ---")
print(df_res.describe(include='all'))  # incl. object para ver recuentos

# Revisar valores únicos de ambiente
print("\nValores únicos de 'ambiente':", df_res['ambiente'].unique())

# Revisar la distribución de los códigos HTTP
print("\nDistribución de 'codigoEstadoHttp':")
print(df_res['codigoEstadoHttp'].value_counts(dropna=False))

# Distribución del tiempo de proceso
print("\nDescripción 'tiempoProceso' (ms):")
print(df_res['tiempoProceso'].describe())

# ---------------------------------------------------------------------
# 4. VISUALIZACIONES
# ---------------------------------------------------------------------
# 4.1 Histograma de tiempo de proceso
plt.hist(df_res['tiempoProceso'].dropna(), bins=20)
plt.title("Distribución del tiempo de proceso (ms)")
plt.xlabel("Tiempo (ms)")
plt.ylabel("Frecuencia")
plt.show()

# 4.2 Gráfico de cajas (Boxplot) para ver outliers de tiempo de proceso
plt.boxplot(df_res['tiempoProceso'].dropna(), vert=False)
plt.title("Boxplot de tiempo de proceso (ms)")
plt.xlabel("Tiempo (ms)")
plt.show()

# ---------------------------------------------------------------------
# 5. EJEMPLOS DE PREGUNTAS Y ANÁLISIS
# ---------------------------------------------------------------------

# A) Tasa de éxito vs. error
total_respuestas = len(df_res)
respuestas_http_ok = df_res[df_res['codigoEstadoHttp'] == 200].shape[0]
respuestas_http_error = df_res[df_res['codigoEstadoHttp'] != 200].shape[0]
print("\nTasa de éxito (código 200): {:.2f}%".format(
    (respuestas_http_ok / total_respuestas)*100 if total_respuestas > 0 else 0
))
print("Cantidad de registros con error (cód no 200):", respuestas_http_error)

# B) Endpoints más comunes
if 'endPoint' in df_res.columns:
    print("\nEndpoints más comunes:")
    print(df_res['endPoint'].value_counts().head(5))

# C) Análisis temporal: Si tenemos 'timestamp_iso' parseado, podemos agrupar por fecha
if pd.api.types.is_datetime64_any_dtype(df_res['timestamp_iso']):
    df_res['fecha'] = df_res['timestamp_iso'].dt.date
    # Conteo diario
    conteo_diario = df_res.groupby('fecha')['app'].count()
    print("\nConteo de logs por fecha (timestamp_iso):")
    print(conteo_diario)

    # Tiempo de proceso promedio por día
    tiempo_promedio_diario = df_res.groupby('fecha')['tiempoProceso'].mean()
    print("\nTiempo de proceso promedio por fecha (ms):")
    print(tiempo_promedio_diario)

    # Podemos visualizarlo (tiempo promedio diario)
    tiempo_promedio_diario.plot(marker='o')
    plt.title("Tiempo de proceso promedio por día")
    plt.xlabel("Fecha")
    plt.ylabel("Tiempo Proceso Promedio (ms)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------
# 6. PARSEO PARCIAL DE LA COLUMNA 'mensaje' (OPCIONAL)
# ---------------------------------------------------------------------
# Ejemplo de parseo parcial (NO siempre funciona dada la alta anidación).
# Se muestra a modo de ejemplo de cómo podríamos extraer data JSON.
import json

def try_parse_json(text):
    """
    Intenta parsear la cadena como JSON.
    Retorna None si falla.
    """
    try:
        return json.loads(text)
    except:
        return None

# Veamos si en 'mensaje' hay algo parseable:
if 'mensaje' in df_res.columns:
    # Ejemplo: parsear 1 fila
    example_msg = df_res['mensaje'].iloc[0]
    parsed = try_parse_json(example_msg)
    print("\nEjemplo parseo JSON en 'mensaje' (primera fila):", parsed)
    # Para parsearlo en masa, se requeriría más limpieza
    # y es posible que no todas las filas contengan JSON válido.

# Fin del Script
print("\n--- Fin del análisis de muestra ---")
