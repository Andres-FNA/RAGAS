

# Sistema RAG Local con Ollama + RAGAS

Este proyecto implementa un sistema RAG (Retrieval-Augmented Generation) completamente local utilizando Ollama, FAISS, embeddings semánticos y evaluación automática con RAGAS, sin depender de OpenAI ni de servicios externos. El objetivo principal es construir un asistente capaz de responder preguntas únicamente con base en documentos institucionales, minimizando alucinaciones y mejorando la precisión mediante recuperación semántica de contexto.

Además del sistema de consulta, el proyecto incluye una evaluación formal del pipeline usando métricas de RAGAS como Faithfulness, Answer Relevancy y Context Precision, permitiendo medir tanto la calidad de las respuestas como la efectividad del sistema de recuperación.

Las evaluaciones se realizan mediante un conjunto de preguntas fijas previamente definidas dentro de `evaluate_rag.py`, lo que permite mantener consistencia entre pruebas, comparar resultados de forma objetiva y analizar mejoras después de realizar ajustes en el sistema.

Al finalizar la evaluación, los resultados se exportan automáticamente a un archivo CSV llamado `resultados_ragas_final.csv`, facilitando el análisis posterior y la elaboración de informes académicos.

---

# ¿Por qué se decidió usar Ollama?

Se decidió utilizar Ollama como motor principal tanto para generación como para evaluación porque permite trabajar con modelos LLM de forma completamente local, lo que resulta especialmente útil en entornos académicos donde no siempre se dispone de presupuesto para APIs pagas como OpenAI.

Otra razón importante fue la privacidad. Al trabajar con sílabos universitarios, documentos académicos y contenido institucional, era fundamental evitar el envío de información a servicios externos. Ollama permite que todo el procesamiento ocurra dentro del equipo local, manteniendo el control completo de los datos.

También se eligió por su facilidad de integración con Python. Su API REST local permite consultar modelos mediante endpoints sencillos como `/api/generate` y `/api/embed`, lo que facilitó la conexión directa con los módulos del proyecto como `vector_store.py`, `rag_engine.py` y `evaluate_rag.py`.

Finalmente, Ollama permitió usar el mismo modelo como generador y como juez en RAGAS. En este proyecto se utilizó `mistral` como LLM generador y también como LLM juez, logrando una evaluación consistente y completamente offline.

---

# Tecnologías utilizadas

El sistema fue desarrollado principalmente en Python 3.10+ y utiliza varias librerías orientadas a recuperación semántica, evaluación automática y procesamiento local de modelos.

FAISS se utilizó para construir la base vectorial y realizar búsquedas rápidas mediante similitud coseno. Pandas permitió organizar y exportar resultados, mientras que NumPy facilitó el manejo de vectores de embeddings.

RAGAS fue utilizado para evaluar el desempeño del sistema mediante métricas especializadas para pipelines RAG. LangChain Community permitió la integración con Ollama tanto para generación como para embeddings. Requests se utilizó para las llamadas HTTP locales y Tabulate para mostrar resultados de configuración de forma estructurada.

También se incorporó `requirements.txt` para facilitar la instalación automática de todas las dependencias necesarias del proyecto.

---

# Archivo requirements.txt

El proyecto incluye un archivo `requirements.txt` que contiene todas las librerías necesarias para ejecutar correctamente el sistema.

Esto permite evitar instalaciones manuales una por una y simplifica mucho la configuración inicial del entorno.

En lugar de instalar cada paquete individualmente, basta con ejecutar:

```bash id="r1"}
pip install -r requirements.txt
```

Esto instalará automáticamente todas las dependencias necesarias para:

* indexación de documentos
* embeddings semánticos
* vectorización con FAISS
* consultas RAG
* evaluación con RAGAS
* exportación de resultados a CSV

Esto hace que el proyecto sea más fácil de replicar y más profesional para entornos académicos y GitHub.

---

# Modelos necesarios en Ollama

Antes de ejecutar el sistema, se deben descargar los modelos que serán utilizados por el pipeline.

El modelo principal de generación es `mistral`, encargado de responder las preguntas del usuario y también de actuar como juez en la evaluación con RAGAS.

Para embeddings semánticos del sistema RAG se utilizó `mxbai-embed-large`, elegido por ofrecer mejor calidad semántica que alternativas más simples como TF-IDF.

Para la evaluación con RAGAS se utilizó `nomic-embed-text`, que funciona correctamente como embedding judge para la métrica de Answer Relevancy.

Por tanto, deben instalarse:

* mistral
* mxbai-embed-large
* nomic-embed-text

---

# Estructura del proyecto

El proyecto está organizado de forma modular para separar claramente cada etapa del pipeline.

La carpeta `docs/` contiene los documentos fuente que serán consultados por el sistema, como los sílabos de Ingeniería de Software y Redes de Comunicación.

La carpeta `vector_db/` almacena el índice FAISS y la metadata generada durante la indexación.

El archivo `document_loader.py` se encarga de cargar documentos y dividirlos en chunks. `vector_store.py` genera embeddings y construye la base vectorial. `rag_engine.py` implementa la lógica principal del sistema RAG y `evaluate_rag.py` ejecuta la evaluación con RAGAS.

El archivo `requirements.txt` centraliza todas las dependencias necesarias para la instalación del proyecto.

Finalmente, `main.py` funciona como punto de entrada principal del sistema.

---

# Explicación de los archivos principales

`document_loader.py` se encarga de cargar archivos `.txt`, `.pdf` y `.docx`, limpiar el contenido textual y dividir los documentos en chunks con overlap para mejorar la recuperación semántica.

`vector_store.py` genera embeddings usando Ollama, normaliza vectores y construye un índice FAISS con similitud coseno real mediante `IndexFlatIP + normalize_L2`. Aquí también se configuran parámetros importantes como el modelo de embeddings y el umbral mínimo de similitud.

`rag_engine.py` representa el corazón del sistema. Recupera chunks relevantes, detecta automáticamente el documento adecuado, construye prompts estrictos con few-shot prompting y consulta el modelo generador evitando alucinaciones cuando no existe suficiente contexto.

`evaluate_rag.py` permite medir formalmente el desempeño del sistema mediante RAGAS. Aquí se definen preguntas fijas de evaluación divididas por categorías: literal, semántico, multi-chunk y control de alucinación.

Estas preguntas no cambian automáticamente, ya que fueron diseñadas para evaluar de forma controlada el comportamiento del sistema y permitir comparaciones entre diferentes versiones del pipeline.

Además, este archivo genera automáticamente:

* tabla de resultados
* análisis crítico por pregunta
* promedios globales
* exportación a CSV

El archivo exportado se guarda como:

```text id="r2"}
resultados_ragas_final.csv
```

Esto facilita la revisión académica y la elaboración de informes.

`main.py` permite ejecutar todo el sistema desde consola, ya sea para indexar documentos, hacer consultas directas o usar el modo interactivo.

---

# Cómo usar el proyecto desde cero

El primer paso consiste en instalar Ollama desde su sitio oficial y dejar corriendo el servicio local mediante `ollama serve`. Este proceso es obligatorio porque todo el sistema depende de la API local de Ollama.

Después se deben descargar los modelos necesarios usando `ollama pull mistral`, `ollama pull mxbai-embed-large` y `ollama pull nomic-embed-text`.

Luego se deben instalar las dependencias del proyecto usando el archivo `requirements.txt`, lo que simplifica completamente la configuración inicial.

Una vez listo el entorno, se debe crear la carpeta `docs/` y colocar allí los documentos que serán utilizados como base de conocimiento. En este caso se trabajó principalmente con sílabos académicos de asignaturas universitarias.

El siguiente paso es ejecutar la indexación mediante:

```bash id="r3"}
python main.py --index
```

Esto carga los documentos, genera chunks, crea embeddings y construye el índice FAISS que luego será usado para recuperación semántica.

Después de indexar, ya pueden hacerse consultas usando:

```bash id="r4"}
python main.py --query "pregunta"
```

o mediante:

```bash id="r5"}
python main.py --interactive
```

que permite múltiples preguntas en una misma sesión.

Finalmente, para medir el desempeño completo del sistema, se ejecuta:

```bash id="r6"}
python evaluate_rag.py
```

Este proceso usa las preguntas fijas definidas en el código, ejecuta la evaluación completa con RAGAS, genera el análisis crítico y exporta automáticamente los resultados al archivo CSV.

---

# Parámetros importantes

Uno de los parámetros más importantes es `chunk_size`, configurado en 400. Este valor define el tamaño de cada fragmento textual y afecta directamente la calidad del retrieval.

El parámetro `overlap`, configurado en 100, permite que exista solapamiento entre chunks consecutivos para evitar pérdida de contexto importante entre fragmentos.

El valor `top_k = 5` define cuántos chunks se recuperan por consulta antes de construir el prompt.

Finalmente, `min_score = 0.70` controla el umbral mínimo de similitud semántica aceptado. Si este valor es demasiado alto, puede perderse contexto útil; si es demasiado bajo, se introduce ruido en la recuperación.

---

# Resultados obtenidos

Durante la evaluación se obtuvieron excelentes resultados en Faithfulness, alcanzando un promedio global de 1.0000. Esto demuestra que el sistema responde con base en evidencia documental real y evita alucinaciones incluso cuando no existe información disponible.

El Context Precision alcanzó 0.7255 como promedio general, aunque este valor se ve afectado por los dos casos de prueba de alucinación. Al excluir esos casos, el promedio sube a 1.0, confirmando que el retrieval con FAISS funciona de forma muy sólida.

La principal debilidad apareció en Answer Relevancy, con un promedio general de 0.3697. Esto ocurre porque el modelo generador todavía tiende a responder con exceso de información, redundancia o mezclando conceptos no solicitados.

Sin contar los casos de alucinación, este valor mejora a 0.4929, lo que demuestra que el problema principal no está en retrieval sino en la fase de generación del LLM.

---

# Conclusión

El proyecto demuestra que es posible construir un sistema RAG robusto, completamente local y sin dependencia de APIs externas, utilizando Ollama, FAISS y RAGAS.

La recuperación semántica mostró un comportamiento sólido y consistente, especialmente gracias al uso de embeddings de alta calidad y similitud coseno real con FAISS.


