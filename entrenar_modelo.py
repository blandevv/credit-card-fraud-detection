"""Entrena el modelo final y exporta los artefactos que consume `app.py`.

Reproduce la etapa de modelamiento del notebook (preparacion de datos, balanceo con
SMOTE y busqueda de hiperparametros con PR-AUC) y guarda en `modelos/`:

- `preprocesador.joblib`: ColumnTransformer (StandardScaler + OneHotEncoding) ajustado
  unicamente sobre el conjunto de entrenamiento.
- `modelo_fraude.joblib`: regresion logistica afinada.
- `metadatos.joblib`: medianas, categorias, nombres de las variables de entrada y metricas
  del modelo en el conjunto de prueba.

Uso: `python entrenar_modelo.py`
"""

import os

import joblib
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import preprocesamiento as pp

RUTA_DATOS = os.path.join('content', 'credit-card-fraud.csv')
RUTA_MODELOS = 'modelos'

RUTA_PREPROCESADOR = os.path.join(RUTA_MODELOS, 'preprocesador.joblib')
RUTA_MODELO = os.path.join(RUTA_MODELOS, 'modelo_fraude.joblib')
RUTA_METADATOS = os.path.join(RUTA_MODELOS, 'metadatos.joblib')
RUTA_PLANTILLA = os.path.join(RUTA_MODELOS, 'plantilla_transacciones.xlsx')

RANDOM_STATE = 42
GRID_LOGISTICA = {'C': [0.01, 0.1, 1, 10]}


def main():
    if not os.path.exists(RUTA_DATOS):
        raise SystemExit(f'No se encontro el dataset en {RUTA_DATOS}')

    df_crudo = pd.read_csv(RUTA_DATOS, encoding='utf-8')

    df, avisos = pp.preparar(df_crudo)
    print(f'Dataset cargado: {df_crudo.shape[0]} filas x {df_crudo.shape[1]} columnas')
    for aviso in avisos:
        print(f'  - {aviso}')
    print(f'Dataset preparado: {df.shape[0]} filas x {df.shape[1]} columnas')
    print(f'Proporcion de fraude: {df_crudo["Class"].mean():.4f}')

    metadatos = pp.calcular_metadatos(df)
    metadatos['categoria_imputada'] = 'Other'
    print(f'\nVariables de entrada requeridas: {", ".join(metadatos["requeridas"])}')
    print(f'Columnas del modelo: {len(metadatos["numericas"])} numericas + '
          f'{len(metadatos["categoricas"])} categoricas')

    y = df_crudo['Class'].astype(int).reset_index(drop=True)

    idx_train, idx_test = train_test_split(
        np.arange(len(df_crudo)), test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    # Cada particion se vuelve a preparar desde el dataset crudo con los metadatos
    # definitivos: es exactamente el camino que recorre la aplicacion, de modo que los
    # artefactos exportados reproducen las metricas reportadas.
    X_train, _ = pp.preparar(df_crudo.iloc[idx_train], metadatos)
    X_test, _ = pp.preparar(df_crudo.iloc[idx_test], metadatos)
    y_train = y.iloc[idx_train].reset_index(drop=True)
    y_test = y.iloc[idx_test].reset_index(drop=True)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    print(f'\nEntrenamiento: {X_train.shape[0]} filas (fraude {y_train.sum()}) | '
          f'Prueba: {X_test.shape[0]} filas (fraude {y_test.sum()})')

    preprocesador = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), metadatos['numericas']),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'),
             metadatos['categoricas']),
        ]
    )

    X_train_t = preprocesador.fit_transform(X_train)
    X_test_t = preprocesador.transform(X_test)
    print(f'Matriz transformada: {X_train_t.shape[1]} columnas -> {list(preprocesador.get_feature_names_out())}')

    X_train_smo, y_train_smo = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train_t, y_train)
    print(f'SMOTE: {X_train_t.shape[0]} -> {X_train_smo.shape[0]} filas '
          f'({int((y_train_smo == 1).sum())} fraudulentos sinteticos)')

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    busqueda = GridSearchCV(
        LogisticRegression(max_iter=1000),
        GRID_LOGISTICA,
        cv=cv,
        scoring='average_precision',
        n_jobs=-1,
    )
    busqueda.fit(X_train_smo, y_train_smo)
    modelo = busqueda.best_estimator_

    print(f'\nMejor PR-AUC (CV) = {busqueda.best_score_:.4f}')
    print(f'Hiperparametros: {busqueda.best_params_}')

    y_pred = modelo.predict(X_test_t)
    y_proba = modelo.predict_proba(X_test_t)[:, 1]

    metricas = {
        'Accuracy': float(accuracy_score(y_test, y_pred)),
        'Precision': float(precision_score(y_test, y_pred)),
        'Recall': float(recall_score(y_test, y_pred)),
        'F1': float(f1_score(y_test, y_pred)),
        'PR-AUC': float(average_precision_score(y_test, y_proba)),
        'ROC-AUC': float(roc_auc_score(y_test, y_proba)),
    }
    matriz = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = matriz.ravel()

    print('\nMetricas en el conjunto de prueba (sin balancear):')
    for nombre, valor in metricas.items():
        print(f'  {nombre:9s} {valor:.4f}')
    print(f'  Verdaderos negativos {tn} | Falsos positivos {fp} | '
          f'Falsos negativos {fn} | Verdaderos positivos {tp}')
    print(f'  Fraudes detectados: {tp} de {tp + fn}')

    coeficientes = sorted(
        zip(preprocesador.get_feature_names_out(), modelo.coef_[0]),
        key=lambda par: abs(par[1]),
        reverse=True,
    )
    print('\nVariables con mayor peso en el modelo:')
    for nombre, peso in coeficientes[:8]:
        print(f'  {nombre:32s} {peso:+.4f}')

    metadatos['nombres_salida'] = list(preprocesador.get_feature_names_out())
    metadatos['metricas'] = metricas
    metadatos['matriz_confusion'] = {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    metadatos['mejores_hiperparametros'] = busqueda.best_params_
    metadatos['pr_auc_cv'] = float(busqueda.best_score_)
    metadatos['variable_objetivo'] = 'Class'
    metadatos['modelo'] = 'Regresion Logistica'
    metadatos['balanceo'] = 'SMOTE'

    os.makedirs(RUTA_MODELOS, exist_ok=True)
    joblib.dump(preprocesador, RUTA_PREPROCESADOR)
    joblib.dump(modelo, RUTA_MODELO)
    joblib.dump(metadatos, RUTA_METADATOS)

    _exportar_plantilla(df_crudo, metadatos)

    print(f'\nArtefactos guardados en {RUTA_MODELOS}/:')
    for ruta in (RUTA_PREPROCESADOR, RUTA_MODELO, RUTA_METADATOS, RUTA_PLANTILLA):
        print(f'  {ruta} ({os.path.getsize(ruta) / 1024:.1f} KB)')

    verificacion = _verificar(df_crudo.iloc[idx_test], y_test)
    print(f'\nVerificacion del pipeline exportado: {verificacion}')


def _exportar_plantilla(df_crudo, metadatos):
    """Guarda un Excel de ejemplo con la estructura minima que espera la app."""
    columnas = ['Time', 'Amount', 'Amount_log', 'MerchantCategory'] + [f'V{i}' for i in range(1, 11)]
    descripcion = [
        'Legítima de madrugada',
        'FRAUDE: monto alto y hora de riesgo',
        'FRAUDE en la banda más dudosa',
        'Legítima que el modelo marca como sospechosa (falso positivo)',
        'Monto corrupto: se corrige con Amount_log',
    ]
    ejemplos = df_crudo.loc[[1503, 1283, 4046, 3410, 1238], columnas].copy()
    ejemplos.insert(0, 'Descripcion', descripcion)
    ejemplos.insert(1, 'Hora', (df_crudo.loc[[1503, 1283, 4046, 3410, 1238], 'Time'] // 3600).values)

    with pd.ExcelWriter(RUTA_PLANTILLA, engine='openpyxl') as writer:
        ejemplos.to_excel(writer, sheet_name='transacciones', index=False)
        pd.DataFrame({
            'Columna': list(metadatos['requeridas']),
            'Obligatoria': ['Sí'] * len(metadatos['requeridas']),
            'Descripcion': [
                *['Componente anónima (PCA original); puede quedar vacía' for _ in range(10)],
                'Segundos desde el inicio del periodo de observación (0–172.740)',
                'Monto de la transacción (>= 0). Si falta o es incoherente se reconstruye desde Amount_log',
                'Categoría del comercio: ' + ', '.join(metadatos['categorias_merchant']),
            ],
        }).to_excel(writer, sheet_name='instrucciones', index=False)


def _verificar(df_prueba_crudo, y_test):
    """Recarga los artefactos desde disco y comprueba que reproducen las metricas.

    Se evalua sobre el dataset **crudo** de prueba, no sobre la version ya preparada:
    es el mismo punto de entrada que usa la aplicacion.
    """
    preprocesador = joblib.load(RUTA_PREPROCESADOR)
    modelo = joblib.load(RUTA_MODELO)
    metadatos = joblib.load(RUTA_METADATOS)

    _, probabilidad, prediccion, _ = pp.predecir(
        preprocesador, modelo, df_prueba_crudo.copy(), metadatos
    )

    metricas = metadatos['metricas']
    recall = float(recall_score(y_test, prediccion))
    precision = float(precision_score(y_test, prediccion, zero_division=0))

    if abs(recall - metricas['Recall']) > 1e-9 or abs(precision - metricas['Precision']) > 1e-9:
        return (f'FALLO | recall {recall:.4f} vs {metricas["Recall"]:.4f} | '
                f'precision {precision:.4f} vs {metricas["Precision"]:.4f}')

    return (f'OK | recall {recall:.4f} | precision {precision:.4f} | '
            f'PR-AUC {average_precision_score(y_test, probabilidad):.4f} | '
            f'coincide con el entrenamiento')


if __name__ == '__main__':
    main()
