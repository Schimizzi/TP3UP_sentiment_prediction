# Análisis de Sentimiento de Tweets con Scikit-Learn y SpaCy

Este proyecto implementa un pipeline completo de Machine Learning para clasificar el sentimiento (positivo o negativo) de tweets, utilizando técnicas clásicas de Procesamiento de Lenguaje Natural (NLP). El modelo final es entrenado, optimizado y guardado para poder realizar predicciones sobre datos nuevos no vistos.

---

## ✨ Características Principales

- **Manejo de Datos a Gran Escala:** Carga eficiente de un subconjunto aleatorio del dataset de 1.6 millones de tweets de Sentiment140 para un desarrollo ágil.
- **Pipeline de Preprocesamiento:** Limpieza de texto robusta que incluye conversión a minúsculas, eliminación de caracteres especiales, lematización con **SpaCy** y eliminación de *stop words*.
- **Vectorización Avanzada:** Uso de **TF-IDF** con **n-gramas** (unigrama y bigrama) para capturar mejor el contexto de las palabras.
- **Optimización de Modelo:** Búsqueda de los mejores hiperparámetros para un modelo de **Regresión Logística** utilizando `GridSearchCV` para mejorar su rendimiento y reducir el sobreajuste.
- **Evaluación y Análisis:** Métricas de rendimiento detalladas con `classification_report` y una **Matriz de Confusión** para visualizar los errores del modelo.
- **Persistencia del Modelo:** El vectorizador y el modelo optimizado se guardan en disco usando `joblib`, permitiendo su reutilización sin necesidad de re-entrenamiento.
- **Inferencia:** Script para cargar el modelo guardado y realizar predicciones sobre un nuevo archivo CSV, exportando los resultados a un archivo Excel.

---

## 🛠️ Tecnologías y Librerías Utilizadas

- Python 3
- Jupyter Notebook
- Pandas
- Scikit-learn
- SpaCy
- WordCloud
- Matplotlib & Seaborn
- Joblib
- Openpyxl

---

## ⚙️ Configuración e Instalación

Para ejecutar este proyecto localmente, sigue estos pasos:

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/tu-usuario/tu-repositorio.git](https://github.com/tu-usuario/tu-repositorio.git)
    cd tu-repositorio
    ```

2.  **Crear un entorno virtual (recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```

3.  **Instalar las dependencias:**
    Crea un archivo `requirements.txt` con el siguiente contenido y luego instálalo.
    ```txt
    pandas
    scikit-learn
    spacy
    matplotlib
    seaborn
    wordcloud
    joblib
    openpyxl
    jupyter
    ```
    Ejecuta el comando de instalación:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Descargar el modelo de SpaCy:**
    ```bash
    python -m spacy download en_core_web_sm
    ```

5.  **Descargar el Dataset:**
    -   Descarga el dataset [Sentiment140](http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip).
    -   Descomprímelo y coloca el archivo `training.1600000.processed.noemoticon.csv` dentro de una carpeta llamada `data/` en la raíz del proyecto.

---

## 🚀 Uso del Proyecto

1.  **Ejecutar el Notebook:** Abre y ejecuta el archivo `NLP_TP3_v1.ipynb` en un entorno de Jupyter.
2.  **Entrenamiento y Guardado:** Al ejecutar todas las celdas, el notebook realizará el preprocesamiento, entrenará el modelo con `GridSearchCV` y guardará los artefactos finales (`vectorizer` y `modelo`) en la carpeta `model/`.
3.  **Realizar Predicciones:** La última celda del notebook está configurada para leer un archivo CSV de prueba (ej. `data/testdata.manual.2009.06.14.csv`), realizar predicciones de sentimiento y guardar los resultados en un archivo llamado `predicciones_muestreo.xlsx`.

---

## 📂 Estructura del Proyecto

```
.
├── data/
│   ├── training.1600000.processed.noemoticon.csv
│   └── testdata.manual.2009.06.14.csv
├── model/
│   ├── schimizzi_modelo_final.joblib
│   └── schimizzi_vectorizer_final.joblib
├── NLP_TP3_v1.ipynb
├── predicciones_muestreo.xlsx
├── requirements.txt
└── README.md
```

---

## 📈 Posibles Mejoras a Futuro

-   **Word Embeddings:** Implementar técnicas como Word2Vec o GloVe para capturar el significado semántico de las palabras.
-   **Modelos de Deep Learning:** Experimentar con redes neuronales recurrentes (LSTM, GRU) para mejorar la comprensión de secuencias de texto.
-   **Modelos de Transformers:** Utilizar modelos pre-entrenados como BERT para obtener un rendimiento de última generación en la clasificación de sentimientos.
-   **Despliegue:** Crear una API (por ejemplo, con Flask o FastAPI) para servir el modelo y permitir predicciones en tiempo real.