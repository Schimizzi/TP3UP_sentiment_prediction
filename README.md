# Análisis de Sentimiento en Tweets: Modelo Propio vs. RoBERTa
Este proyecto desarrolla y evalúa un modelo de análisis de sentimiento para tuits en inglés. Se entrena un modelo clásico de Regresión Logística con TF-IDF y se compara su rendimiento con un modelo de vanguardia pre-entrenado (RoBERTa) de Hugging Face, especializado en el análisis de sentimiento de tuits.

## Descripción del Proyecto
El objetivo principal es construir una herramienta práctica para el análisis de sentimiento, cubriendo todo el ciclo de vida de un proyecto de Machine Learning:

**Análisis Exploratorio de Datos (EDA)**: Se analizan metadatos como la longitud de los tuits, el uso de menciones (@), hashtags (#) y palabras en mayúsculas para entender las características del conjunto de datos.

**Preprocesamiento de Texto**: Se aplica un pipeline de limpieza que incluye la conversión a minúsculas, eliminación de caracteres no alfabéticos, lematización y eliminación de stopwords con la librería spaCy.

**Entrenamiento y Optimización**: Se entrena un modelo de Regresión Logística sobre vectores TF-IDF (n-gramas de 1 y 2) y se optimizan sus hiperparámetros usando GridSearchCV.

**Comparación de Modelos**: El rendimiento del modelo local se compara cualitativamente con las predicciones de RoBERTa (cardiffnlp/twitter-roberta-base-sentiment), un modelo de Transformers especializado en tuits.

**Predicción y Exportación**: El modelo entrenado se utiliza para realizar predicciones sobre un conjunto de datos nuevo y los resultados se exportan a un archivo Excel para su fácil revisión.

## Modelos Utilizados
1. **Modelo Local**: Regresión Logística con TF-IDF
Algoritmo: Regresión Logística.

Vectorización: TF-IDF con n-gramas (1, 2) y un máximo de 20,000 características.

Capacidades: Clasificación binaria (Positivo/Negativo).

Entrenamiento: Realizado localmente sobre un conjunto de 1.6 millones de tuits.

2. **Modelo de Comparación**: RoBERTa para Sentimiento en Tuits
Modelo: cardiffnlp/twitter-roberta-base-sentiment de Hugging Face.

Arquitectura: Modelo de Transformers (RoBERTa) pre-entrenado y ajustado específicamente para el lenguaje de los tuits.

Capacidades: Clasificación multiclase (Positivo, Negativo, Neutral), con una mejor comprensión del contexto, el sarcasmo y los matices del lenguaje.

# Estructura del Proyecto
```
├── 📂 data/
│   ├── training.1600000.processed.noemoticon.csv   # Dataset de entrenamiento
│   └── testdata.manual.2009.06.14.csv              # Dataset para predicción/comparación
├── 📂 data_processed/
│   └── split_data_cleaned_1.6kk.joblib             # Datos limpios y divididos (generado)
├── 📂 model_tf_idf/
│   ├── schimizzi_modelo_1.6kk.joblib               # Modelo de Regresión Logística entrenado (generado)
│   └── schimizzi_vectorizer_1.6kk.joblib           # Vectorizador TF-IDF ajustado (generado)
├── 📂 predict/
│   └── nueva_prediccion.xlsx                       # Salida con las predicciones (generado)
├── 📜 modelo_TF-IDF_sent_v2.3.ipynb                # Notebook para EDA, entrenamiento y evaluación del modelo local.
├── 📜 myModel_vs_roBERTa.ipynb                     # Notebook para comparar el modelo local con RoBERTa.
└── 📜 requirements.txt                             # Dependencias del proyecto.
```

# Instalación y Configuración

Para ejecutar este proyecto, sigue los siguientes pasos:

Clona el repositorio:
```Bash
git clone https://github.com/Schimizzi/TP3UP_sentiment_prediction.git
cd TP3UP_sentiment_prediction
```

Crea un entorno virtual (recomendado para Windows):
```Bash
python -m venv venv
.\venv\Scripts\activate
```

Instala las dependencias usando el archivo requirements.txt:
```Bash
pip install -r requirements.txt
```

Descarga el modelo de lenguaje de spaCy: Este modelo es necesario para la limpieza y lematización del texto.
```Bash
python -m spacy download en_core_web_sm
```

# Uso del Proyecto

El proyecto está dividido en dos notebooks principales:

## 1. **Entrenamiento del Modelo Local** (modelo_TF-IDF_sent_v2.3.ipynb)

Este notebook te guiará a través de:

La carga y el análisis exploratorio del dataset de 1.6 millones de tuits.

El proceso de limpieza y preprocesamiento de texto.

La vectorización TF-IDF.

El entrenamiento del modelo de Regresión Logística con GridSearchCV para encontrar los mejores parámetros.

La evaluación del modelo final con reportes de clasificación y una matriz de confusión.

El guardado del modelo y el vectorizador en la carpeta model_tf_idf/.

## 2. **Comparación y Predicción** (myModel_vs_roBERTa.ipynb)
Este notebook se enfoca en:

Cargar el modelo local previamente guardado.

Cargar el modelo RoBERTa de Hugging Face.

Realizar predicciones sobre un conjunto tuits de prueba con ambos modelos.

Mostrar una tabla comparativa para analizar las diferencias en las predicciones.

Guardar las predicciones del modelo local sobre nuevos datos en un archivo Excel en la carpeta predict/.

# Análisis de Resultados
La comparación entre el modelo local y RoBERTa arrojó las siguientes conclusiones:

Manejo del Contexto: RoBERTa, al ser un Transformer, es superior en la comprensión de matices, sarcasmo y el contexto general de un tuit.

Clasificación de Neutralidad: El modelo local solo clasifica en "Positivo" y "Negativo", mientras que RoBERTa puede identificar tuits "Neutrales", lo que le otorga mayor precisión en textos sin una carga sentimental clara.

Confianza en la Predicción: El modelo de Hugging Face proporciona un puntaje de confianza, que tiende a ser más bajo en textos ambiguos, ofreciendo una capa adicional de interpretabilidad.

**A pesar de las ventajas del modelo pre-entrenado, el modelo de Regresión Logística local logró un rendimiento notable y satisfactorio, demostrando ser una solución eficaz y bien construida.**