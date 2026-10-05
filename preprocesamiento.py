"""Feature engineering del proyecto de deteccion de fraude.

Este modulo es el unico lugar donde se define como se prepara una transaccion para el
modelo. Lo usan tanto `entrenar_modelo.py` como `app.py`, de modo que la aplicacion no
puede divergir del entrenamiento: cualquier cambio aqui afecta a ambos de forma identica.
"""

import numpy as np
import pandas as pd

VARIABLES_IDENTIFICADORAS = [
    'RecordID',
    'FullName',
    'Phone',
    'ZodiacSign',
    'FavoriteColor',
    'Hobby',
    'DeviceType',
]

VARIABLES_ENTRADA = [
    'V1', 'V2', 'V3', 'V4', 'V5',
    'V6', 'V7', 'V8', 'V9', 'V10',
    'Time', 'Amount', 'MerchantCategory',
]

VARIABLES_NUMERICAS = [
    'V1', 'V2', 'V3', 'V4', 'V5',
    'V6', 'V7', 'V8', 'V9', 'V10',
    'Time', 'Amount',
    'V1_missing', 'V3_missing',
    'Hora',
]

VARIABLES_CATEGORICAS = ['MerchantCategory', 'Franja']

VARIABLES_IMPUTADAS = ['V1', 'V3']

CATEGORIAS_MERCHANT = ['Restaurant', 'Gas', 'Grocery', 'Online', 'Travel', 'Other']

ETIQUETAS_FRANJA = [
    'Madrugada (00-05)',
    'Mañana (06-11)',
    'Tarde (12-17)',
    'Noche (18-23)',
]

CORTES_FRANJA = [-1, 5, 11, 17, 23]

TOLERANCIA_AMOUNT = 0.05

UMBRAL_PREDICCION = 0.5


def _metadatos_provisionales(df):
    """Metadatos estimados desde los propios datos.

    Solo se usa durante el entrenamiento: las medianas y categorias definitivas se
    recalculan despues con `calcular_metadatos` sobre el dataset ya preparado.
    """
    numericas = [c for c in VARIABLES_NUMERICAS if c in df.columns]
    return {
        'numericas': list(VARIABLES_NUMERICAS),
        'categoricas': list(VARIABLES_CATEGORICAS),
        'requeridas': list(VARIABLES_ENTRADA),
        'medianas': {c: float(pd.to_numeric(df[c], errors='coerce').median()) for c in numericas},
        'categorias': {
            'MerchantCategory': list(CATEGORIAS_MERCHANT),
            'Franja': list(ETIQUETAS_FRANJA),
        },
    }


def calcular_metadatos(df_preparado):
    """Deriva de los datos de entrenamiento los parametros que consume la aplicacion.

    Se calcula sobre el dataset completo, igual que en el notebook, para que el modelo
    exportado y los metadatos de la app provengan del mismo ajuste.
    """
    categorias = {}
    for col in VARIABLES_CATEGORICAS:
        valores = df_preparado[col].astype(object)
        categorias[col] = [v for v in pd.unique(valores) if not pd.isna(v)]

    medianas = {col: float(df_preparado[col].median()) for col in VARIABLES_NUMERICAS}

    percentiles = df_preparado[VARIABLES_NUMERICAS].quantile([0.005, 0.5, 0.995])
    rangos = {
        col: {
            'min': float(np.floor(percentiles.loc[0.005, col] * 100) / 100),
            'max': float(np.ceil(percentiles.loc[0.995, col] * 100) / 100),
            'mediana': float(percentiles.loc[0.5, col]),
        }
        for col in VARIABLES_NUMERICAS
    }

    return {
        'numericas': list(VARIABLES_NUMERICAS),
        'categoricas': list(VARIABLES_CATEGORICAS),
        'requeridas': list(VARIABLES_ENTRADA),
        'medianas': medianas,
        'categorias': categorias,
        'categorias_merchant': list(CATEGORIAS_MERCHANT),
        'etiquetas_franja': list(ETIQUETAS_FRANJA),
        'rangos': rangos,
        'umbral': UMBRAL_PREDICCION,
    }


def reconstruir_amount(df, avisos):
    """Usa `Amount_log` como fuente de verdad cuando `Amount` es nulo o incoherente.

    `Amount_log = log(1 + Amount)`, por lo que la reconstruccion es exacta. Se replica el
    criterio de integridad del notebook: si el monto no se reconstruye con tolerancia
    0.05, el valor corrupto es `Amount` y se descarta.
    """
    if 'Amount_log' not in df.columns:
        if 'Amount' not in df.columns:
            raise ValueError("Se requiere 'Amount' o 'Amount_log' para determinar el monto.")
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        return df

    df['Amount_log'] = pd.to_numeric(df['Amount_log'], errors='coerce')
    reconstruido = np.expm1(df['Amount_log'])

    if 'Amount' in df.columns:
        monto = pd.to_numeric(df['Amount'], errors='coerce')
        incoherente = (monto - reconstruido).abs() > TOLERANCIA_AMOUNT
        n_incoherentes = int(incoherente.sum())
        monto = monto.mask(incoherente, reconstruido)
    else:
        n_incoherentes = 0
        monto = reconstruido

    nulos = int(monto.isna().sum())
    monto = monto.fillna(reconstruido).fillna(0.0)

    if n_incoherentes:
        avisos.append(f'{n_incoherentes} monto(s) incoherente(s) reconstruido(s) desde Amount_log.')
    if nulos:
        avisos.append(f'{nulos} monto(s) nulo(s) reconstruido(s) desde Amount_log.')

    df['Amount'] = monto.clip(lower=0.0)
    return df.drop(columns=['Amount_log'])


def derivar_hora_franja(df):
    """Deriva `Hora` y `Franja` a partir de `Time` (segundos desde el inicio del periodo).

    Los cortes son los del notebook: la franja solo aplica a las primeras 24 horas del
    periodo, por lo que las transacciones del segundo dia quedan sin franja, valor que el
    codificador trata como una categoria propia.
    """
    df['Time'] = pd.to_numeric(df['Time'], errors='coerce').fillna(0).clip(lower=0)
    df['Hora'] = (df['Time'] // 3600).astype(int)
    franja = pd.cut(df['Hora'], bins=CORTES_FRANJA, labels=ETIQUETAS_FRANJA, right=True)
    df['Franja'] = pd.Categorical(franja, categories=ETIQUETAS_FRANJA)
    return df


def imputar_nulos(df, metadatos, avisos):
    """Imputa V1 y V3 con la mediana del entrenamiento y crea los indicadores de faltante."""
    for col in VARIABLES_IMPUTADAS:
        bandera = f'{col}_missing'
        df[bandera] = df[col].isna().astype(int)

        nulos = int(df[col].isna().sum())
        mediana = float(metadatos['medianas'].get(col, 0.0))
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(mediana)
        if nulos:
            avisos.append(f'{nulos} valor(es) nulo(s) en {col} imputado(s) con la mediana ({mediana:.4f}).')
    return df


def normalizar_categoricas(df, metadatos, avisos):
    """Imputa `MerchantCategory` con 'Other' y fija las categorias en el dtype esperado."""
    conocidas = metadatos['categorias']['MerchantCategory']
    imputacion = metadatos.get('categoria_imputada', 'Other')

    if 'MerchantCategory' not in df.columns:
        df['MerchantCategory'] = imputacion

    df['MerchantCategory'] = df['MerchantCategory'].astype(object).replace(r'^\s*$', np.nan, regex=True)
    nulos = int(df['MerchantCategory'].isna().sum())
    df['MerchantCategory'] = df['MerchantCategory'].fillna(imputacion)
    if nulos:
        avisos.append(f'{nulos} valor(es) nulo(s) en MerchantCategory imputado(s) con "{imputacion}".')

    desconocidas = df['MerchantCategory'].notna() & ~df['MerchantCategory'].isin(conocidas)
    n_desconocidas = int(desconocidas.sum())
    if n_desconocidas:
        df.loc[desconocidas, 'MerchantCategory'] = imputacion
        avisos.append(f'{n_desconocidas} categoría(s) no vistas en el entrenamiento mapeada(s) a "{imputacion}".')

    df['MerchantCategory'] = pd.Categorical(df['MerchantCategory'], categories=conocidas)

    etiquetas = [e for e in metadatos['categorias']['Franja'] if not pd.isna(e)]
    df['Franja'] = pd.Categorical(df['Franja'], categories=etiquetas)
    return df


def preparar(df, metadatos=None):
    """Aplica todo el preprocesamiento y devuelve `(df_preparado, avisos)`.

    `df` puede traer columnas de sobra (identificadores, `Class`, `Amount_log`): se
    ignoran. Las columnas ausentes se rellenan con la mediana del entrenamiento para que
    la aplicacion nunca falle por un archivo incompleto. Sin `metadatos` (entrenamiento),
    las medianas y categorias se estiman desde los propios datos.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Se esperaba un DataFrame de pandas.')

    avisos = []
    df = df.copy()

    if metadatos is None:
        metadatos = _metadatos_provisionales(df)

    a_descartar = VARIABLES_IDENTIFICADORAS + [
        'Class', 'Hora', 'Franja', 'V1_missing', 'V3_missing',
    ]
    df = df.drop(columns=[c for c in a_descartar if c in df.columns])

    faltantes = [c for c in VARIABLES_ENTRADA if c not in df.columns]
    if faltantes:
        avisos.append(f"Columnas ausentes en la entrada, rellenadas con valores neutros: {', '.join(faltantes)}.")

    for col in VARIABLES_ENTRADA:
        if col not in df.columns:
            df[col] = np.nan

    df = reconstruir_amount(df, avisos)
    df = imputar_nulos(df, metadatos, avisos)
    df = derivar_hora_franja(df)
    df = normalizar_categoricas(df, metadatos, avisos)

    for col in VARIABLES_NUMERICAS:
        valores = pd.to_numeric(df[col], errors='coerce')
        no_numericos = int(valores.isna().sum())
        if no_numericos and col not in VARIABLES_IMPUTADAS:
            avisos.append(f'{no_numericos} valor(es) no numérico(s) en {col} reemplazado(s) por la mediana.')
        df[col] = valores.fillna(metadatos['medianas'].get(col, 0.0))

    return df[VARIABLES_NUMERICAS + VARIABLES_CATEGORICAS], avisos


def predecir(preprocesador, modelo, df, metadatos, umbral=None):
    """Pipeline completo: preparar -> escalar y codificar -> predecir.

    Devuelve `(df_preparado, probabilidad_fraude, prediccion, avisos)`.
    """
    umbral = metadatos['umbral'] if umbral is None else umbral
    df_preparado, avisos = preparar(df, metadatos)
    matriz = preprocesador.transform(df_preparado)
    probabilidad = modelo.predict_proba(matriz)[:, 1]
    prediccion = (probabilidad >= umbral).astype(int)
    return df_preparado, probabilidad, prediccion, avisos
