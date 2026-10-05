import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from streamlit.runtime import exists as runtime_activo

if not runtime_activo():
    raise SystemExit(
        'Esta aplicacion es de Streamlit y debe iniciarse con:\n'
        '    streamlit run app.py\n'
        'Ejecutar `python app.py` la deja en "bare mode" y produce cientos de avisos.'
    )

import preprocesamiento as pp

RUTA_MODELOS = 'modelos'
RUTA_PREPROCESADOR = os.path.join(RUTA_MODELOS, 'preprocesador.joblib')
RUTA_MODELO = os.path.join(RUTA_MODELOS, 'modelo_fraude.joblib')
RUTA_METADATOS = os.path.join(RUTA_MODELOS, 'metadatos.joblib')
RUTA_PLANTILLA = os.path.join(RUTA_MODELOS, 'plantilla_transacciones.xlsx')

st.set_page_config(page_title='Detección de fraude con tarjeta de crédito', page_icon='💳', layout='wide')

st.title('Detección de fraude en transacciones con tarjeta de crédito')
st.write(
    'Aplicación de despliegue del proyecto: recibe las variables de una transacción en el '
    'momento de la autorización y decide si debe **aprobarse** o **bloquearse**. El modelo es '
    'una regresión logística entrenada con el pipeline del notebook `notebooks/main.ipynb`.'
)


# Cargar los artefactos del entrenamiento una sola vez
@st.cache_resource
def load_resources():
    preprocesador = joblib.load(RUTA_PREPROCESADOR)
    modelo = joblib.load(RUTA_MODELO)
    metadatos = joblib.load(RUTA_METADATOS)
    return preprocesador, modelo, metadatos


try:
    preprocesador, modelo, metadatos = load_resources()
except FileNotFoundError:
    st.error(
        f'No se encontraron los artefactos en `{RUTA_MODELOS}/`. '
        'Ejecuta `python entrenar_modelo.py` para generarlos.'
    )
    st.stop()
except Exception as e:
    st.error(f'Error al cargar el modelo pre-entrenado: {e}')
    st.stop()


def aportes_al_log_odds(fila_preparada):
    """Aporte de cada variable a la decisión, para explicarla en términos de negocio.

    En una regresión logística el aporte es `coeficiente * valor`: en las variables
    numéricas el valor ya estandarizado y en las categóricas el indicador one-hot. La suma
    de todos los aportes es el log-odds de la predicción, de modo que un aporte positivo
    empuja la transacción hacia fraude y uno negativo hacia legítima.
    """
    matriz = preprocesador.transform(fila_preparada)
    coeficientes = modelo.coef_[0]
    aporte = {}

    for i, columna in enumerate(metadatos['numericas']):
        aporte[columna] = float(coeficientes[i] * matriz[0, i])

    codificador = preprocesador.named_transformers_['cat']
    inicio = preprocesador.output_indices_['cat'].start
    desplazamiento = inicio
    for j, columna in enumerate(metadatos['categoricas']):
        categorias = list(codificador.categories_[j])
        categorias = categorias[1:] if codificador.drop == 'first' else categorias
        for k, categoria in enumerate(categorias):
            indice = desplazamiento + k
            etiqueta = '(sin valor)' if pd.isna(categoria) else str(categoria)
            aporte[f'{columna} = {etiqueta}'] = float(coeficientes[indice] * matriz[0, indice])
        desplazamiento += len(categorias)

    aporte['Sesgo del modelo'] = float(modelo.intercept_[0])
    return pd.Series(aporte).sort_values(key=np.abs, ascending=False)


def nivel_riesgo(probabilidad):
    if probabilidad >= 0.90:
        return '🔴 Crítico'
    if probabilidad >= 0.50:
        return '🟠 Alto'
    if probabilidad >= 0.20:
        return '🟡 Medio'
    return '🟢 Bajo'


with st.sidebar:
    st.header('Modelo en producción')
    st.markdown(f"**{metadatos['modelo']}**")
    st.caption(
        f"Entrenado con {metadatos['balanceo']} sobre el 70% del dataset y evaluado sobre el "
        '30% de prueba, sin balancear (tasa real de fraude 3%).'
    )

    metricas = metadatos['metricas']
    izquierda, derecha = st.columns(2)
    izquierda.metric('PR-AUC', f"{metricas['PR-AUC']:.4f}")
    derecha.metric('ROC-AUC', f"{metricas['ROC-AUC']:.4f}")
    izquierda.metric('Recall', f"{metricas['Recall']:.4f}")
    derecha.metric('Precision', f"{metricas['Precision']:.4f}")
    matriz_confusion = metadatos['matriz_confusion']
    st.caption(
        f"Recall 100% = {matriz_confusion['tp']} de "
        f"{matriz_confusion['tp'] + matriz_confusion['fn']} fraudes detectados en el conjunto de prueba."
    )

    st.divider()
    st.subheader('Umbral de decisión')
    umbral = st.slider(
        'Probabilidad mínima para marcar la transacción como fraude',
        min_value=0.05,
        max_value=0.95,
        value=float(metadatos['umbral']),
        step=0.05,
    )
    st.caption(
        'Un umbral menor genera más alertas a costa de más falsos positivos. En detección de '
        'fraude se prioriza el recall porque el costo de dejar pasar un fraude es mayor.'
    )

    with st.expander('Variables que recibe el modelo'):
        st.write(f"{len(metadatos['numericas'])} numéricas + {len(metadatos['categoricas'])} categóricas")
        st.code('\n'.join(metadatos['requeridas']), language=None)

tab_individual, tab_lote = st.tabs(['Transacción individual', 'Lote de transacciones'])

with tab_individual:
    st.header('Datos de la transacción')
    st.caption(
        'Los valores por defecto son la mediana del dataset de entrenamiento. Deja `V1` o `V3` '
        'vacíos para simular un dato faltante: la app lo imputa con la mediana, igual que el notebook.'
    )

    rangos = metadatos['rangos']
    componentes_senal = ['V1', 'V2', 'V3', 'V4', 'V5']
    componentes_ruido = ['V6', 'V7', 'V8', 'V9', 'V10']
    valores = {}

    with st.form('form_transaccion'):
        st.markdown('**Componentes V1–V5, donde se concentra la señal de fraude**')
        for columna, caja in zip(componentes_senal, st.columns(5)):
            rango = rangos[columna]
            admiten_nulo = columna in ('V1', 'V3')
            valores[columna] = caja.number_input(
                columna,
                min_value=rango['min'],
                max_value=rango['max'],
                value=None if admiten_nulo else rango['mediana'],
                step=0.01,
                key=f'entrada_{columna}',
            )

        st.markdown('**Componentes V6–V10, sin poder discriminante**')
        for columna, caja in zip(componentes_ruido, st.columns(5)):
            rango = rangos[columna]
            valores[columna] = caja.number_input(
                columna,
                min_value=rango['min'],
                max_value=rango['max'],
                value=rango['mediana'],
                step=0.01,
                key=f'entrada_{columna}',
            )

        st.markdown('**Monto y contexto**')
        monto, hora, minuto, comercio = st.columns(4)
        valores['Amount'] = monto.number_input(
            'Amount (monto)',
            min_value=0.0,
            max_value=2000.0,
            value=34.8,
            step=0.5,
            key='entrada_amount',
        )
        hora_ingresada = hora.number_input(
            'Hora del periodo (0–47)', min_value=0, max_value=47, value=12, step=1, key='entrada_hora'
        )
        minuto_ingresado = minuto.number_input(
            'Minuto', min_value=0, max_value=59, value=0, step=1, key='entrada_minuto'
        )
        valores['MerchantCategory'] = comercio.selectbox(
            'MerchantCategory', metadatos['categorias_merchant'], key='entrada_merchant'
        )
        enviar = st.form_submit_button('Analizar transacción', type='primary')

    if enviar:
        valores['Time'] = hora_ingresada * 3600 + minuto_ingresado * 60
        st.session_state['transaccion'] = dict(valores)

    if st.session_state.get('transaccion') and st.button('Analizar otra transacción'):
        st.session_state.pop('transaccion')
        st.rerun()

    if st.session_state.get('transaccion'):
        valores = st.session_state['transaccion']
        hora_ingresada = valores['Time'] // 3600
        st.caption(
            f"`Time` = {valores['Time']} s · `Hora` = {hora_ingresada} · "
            + ('fuera de las cuatro franjas (segundo día del periodo)'
               if hora_ingresada > 23 else
               f"franja = {[f for f, b in zip(metadatos['etiquetas_franja'], [5, 11, 17, 23]) if b >= hora_ingresada][0]}")
        )

        with st.spinner('Evaluando el modelo...'):
            preparado, probabilidad, prediccion, avisos = pp.predecir(
                preprocesador, modelo, pd.DataFrame([valores]), metadatos, umbral
            )

        probabilidad_fraude = float(probabilidad[0])
        fraude = int(prediccion[0]) == 1

        if fraude and probabilidad_fraude >= 0.90:
            st.error(f'### 🚫 Transacción marcada como FRAUDE\nProbabilidad de fraude: **{probabilidad_fraude:.2%}**')
        elif fraude:
            st.warning(f'### ⚠️ Transacción marcada como FRAUDE\nProbabilidad de fraude: **{probabilidad_fraude:.2%}**')
        else:
            st.success(f'### ✅ Transacción legítima\nProbabilidad de fraude: **{probabilidad_fraude:.2%}**')

        c1, c2, c3 = st.columns(3)
        c1.metric('Clasificación', 'FRAUDE' if fraude else 'LEGÍTIMA')
        c2.metric('Nivel de riesgo', nivel_riesgo(probabilidad_fraude))
        c3.metric('Umbral aplicado', f'{umbral:.2f}')
        st.progress(probabilidad_fraude, text=f'Probabilidad de fraude {probabilidad_fraude:.2%}')

        if fraude:
            st.caption(
                'El umbral está por debajo de la probabilidad estimada, por lo que la app '
                'recomienda **bloquear la transacción y revisar el caso**.'
            )

        if avisos:
            with st.expander('Ajustes aplicados por el preprocesamiento'):
                for aviso in avisos:
                    st.write(f'- {aviso}')

        with st.expander('¿Por qué esta decisión?'):
            aporte = aportes_al_log_odds(preparado)
            st.dataframe(aporte.rename('Aporte al log-odds').to_frame().style.format('{:.4f}'))
            st.caption(
                'Un aporte positivo empuja la transacción hacia fraude y uno negativo hacia '
                'legítima. Los valores numéricos están estandarizados, por lo que los pesos son '
                'comparables entre variables.'
            )

        with st.expander('Variables que recibió el modelo'):
            st.dataframe(preparado.T.astype(str).rename(columns={0: 'Valor'}))
    else:
        st.info('Completa el formulario y presiona **Analizar transacción**.')
        st.caption('La decisión se recalcula sola si cambias el umbral en la barra lateral.')

with tab_lote:
    st.header('Predicción por lotes')
    st.caption(
        'El archivo debe tener una fila por transacción y las columnas `V1`–`V10`, `Time`, '
        '`Amount` y `MerchantCategory`. Las columnas adicionales se ignoran. Si incluye la '
        'columna `Class`, la app también calcula las métricas de desempeño sobre ese archivo.'
    )

    if os.path.exists(RUTA_PLANTILLA):
        with open(RUTA_PLANTILLA, 'rb') as plantilla:
            st.download_button(
                'Descargar plantilla de ejemplo (.xlsx)',
                plantilla,
                file_name='plantilla_transacciones.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            )

    archivo = st.file_uploader('Sube un archivo Excel o CSV con las transacciones',
                               type=['xlsx', 'xls', 'csv'])

    if archivo is not None:
        try:
            if archivo.name.lower().endswith('.csv'):
                df_lote = pd.read_csv(archivo)
            else:
                df_lote = pd.read_excel(archivo)
        except Exception as e:
            st.error(f'No se pudo leer el archivo: {e}')
            df_lote = None

        if df_lote is not None:
            st.write(f'**{len(df_lote)} transacciones leídas.** Vista previa:')
            st.dataframe(df_lote.head(10))

            if st.button('Analizar lote', type='primary'):
                with st.spinner('Procesando y prediciendo...'):
                    _, probabilidad, prediccion, avisos = pp.predecir(
                        preprocesador, modelo, df_lote, metadatos, umbral
                    )

                resultados = df_lote.copy()
                resultados['Probabilidad_Fraude'] = probabilidad
                resultados['Clasificacion'] = np.where(prediccion == 1, 'FRAUDE', 'LEGÍTIMA')

                if avisos:
                    with st.expander('Ajustes aplicados por el preprocesamiento'):
                        for aviso in avisos:
                            st.write(f'- {aviso}')

                if metadatos['variable_objetivo'] in resultados.columns:
                    y_real = resultados[metadatos['variable_objetivo']].astype(int)
                    matriz = confusion_matrix(y_real, prediccion, labels=[0, 1])

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric('Transacciones', f'{len(resultados)}')
                    c2.metric('Marcadas como fraude', f'{int(prediccion.sum())} ({prediccion.mean():.2%})')
                    c3.metric('Recall', f'{recall_score(y_real, prediccion, zero_division=0):.4f}')
                    c4.metric('Precision', f'{precision_score(y_real, prediccion, zero_division=0):.4f}')

                    st.write('Matriz de confusión (filas = real, columnas = predicho):')
                    st.dataframe(
                        pd.DataFrame(matriz,
                                     index=['Real 0 (Legítima)', 'Real 1 (Fraude)'],
                                     columns=['Predicho 0', 'Predicho 1']),
                    )
                    st.caption(
                        f"Accuracy {accuracy_score(y_real, prediccion):.4f} · "
                        f"F1 {f1_score(y_real, prediccion, zero_division=0):.4f} · "
                        f"{int(matriz.sum() - np.trace(matriz))} errores de {len(resultados)}"
                    )

                st.subheader('Resultados')
                st.dataframe(resultados)

                st.download_button(
                    'Descargar resultados (.csv)',
                    resultados.to_csv(index=False).encode('utf-8'),
                    file_name='resultados_deteccion_fraude.csv',
                    mime='text/csv',
                )
    else:
        st.info('Sube un archivo para comenzar.')
