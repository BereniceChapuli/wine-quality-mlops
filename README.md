# Wine Quality — MLOps con MLflow

**Proyecto individual | Optimización de hiperparámetros y reproducibilidad**

**Autora:** Berenice Chapuli  
**GitHub:** [https://github.com/BereniceChapuli](https://github.com/BereniceChapuli)  
**Repositorio:** [https://github.com/BereniceChapuli/wine-quality-mlops](https://github.com/BereniceChapuli/wine-quality-mlops)

## 1. Descripción del proyecto

Este proyecto implementa un flujo reproducible de entrenamiento de Machine Learning para clasificar la calidad del vino tinto mediante regresión logística.

Se utiliza **Scikit-learn** para el preprocesamiento, entrenamiento y evaluación del modelo, y **MLflow Tracking** para registrar los parámetros, métricas, artefactos y modelos generados durante los experimentos.

El objetivo es evaluar el impacto de diferentes configuraciones de hiperparámetros, comparar los resultados y comprobar la reproducibilidad del entrenamiento.

## 2. Dataset

Se utiliza el conjunto de datos `winequality-red.csv`, que contiene 1,599 registros originales y 11 variables predictoras relacionadas con las características fisicoquímicas del vino.

Durante el preprocesamiento:

- Se eliminan registros duplicados, conservando 1,359 observaciones.
- Se genera la variable binaria `quality_class`:
	- Clase 0: calidad inferior a 6.
	- Clase 1: calidad igual o superior a 6.
- Se divide el conjunto de datos en 80% entrenamiento y 20% prueba.
- Se aplica `StandardScaler` para estandarizar las variables predictoras.
- Se utiliza la semilla aleatoria `42` para mantener la consistencia de los experimentos.

## 3. Estructura del proyecto

```
wine-quality-mlops/
├── train.py
├── requirements.txt
├── README.md
├── data/
│   └── winequality-red.csv
└── artifacts/
		└── matriz_confusion.png
```

El archivo `train.py` contiene las funciones para cargar los datos, prepararlos, entrenar el modelo, evaluar sus resultados y registrar el experimento en MLflow.

El notebook original se conserva únicamente como referencia del análisis exploratorio.

## 4. Tecnologías utilizadas

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- MLflow
- Git y GitHub
- Visual Studio Code
- Ubuntu mediante WSL

Las versiones de las dependencias utilizadas se encuentran documentadas en `requirements.txt`.

## 5. Instalación y configuración

### Clonar el repositorio

```bash
git clone https://github.com/BereniceChapuli/wine-quality-mlops.git
cd wine-quality-mlops
```

### Crear un entorno virtual

En Ubuntu o WSL:

```bash
python -m venv .venv
source .venv/bin/activate
```

### Instalar dependencias

```bash
python -m pip install -r requirements.txt
```

Se recomienda utilizar una versión de Python compatible con las dependencias registradas.

## 6. Ejecución del entrenamiento

Para ejecutar el modelo con los hiperparámetros de referencia:

```bash
python train.py --C 1.0 --max_iter 100 --tol 0.0001 --seed 42
```

### Parámetros disponibles

| Parámetro | Descripción | Valor predeterminado |
| --- | --- | --- |
| `--C` | Inverso de la intensidad de regularización | `1.0` |
| `--max_iter` | Número máximo de iteraciones | `100` |
| `--tol` | Tolerancia de convergencia | `0.0001` |
| `--seed` | Semilla aleatoria | `42` |
| `--data` | Ruta del archivo de datos | `data/winequality-red.csv` |

Para consultar la ayuda:

```bash
python train.py --help
```

## 7. Seguimiento de experimentos con MLflow

Los experimentos se registran bajo el nombre:

**Wine_Quality_Hyperparameters**

Cada corrida almacena:

- Hiperparámetros de entrenamiento.
- Semilla aleatoria.
- Accuracy.
- Precision.
- Recall.
- F1-score.
- Matriz de confusión como artefacto.
- Modelo de regresión logística entrenado.

### Configuración utilizada durante el desarrollo

El proyecto se ejecutó en Ubuntu mediante WSL, utilizando una base de datos SQLite de MLflow ubicada en:

```
/home/berenicechx/mlflow.db
```

Para visualizar los experimentos en el entorno original:

```bash
mlflow ui --backend-store-uri sqlite:////home/berenicechx/mlflow.db --port 5001
```

Abrir en el navegador:

[http://127.0.0.1:5001](http://127.0.0.1:5001/)

**Nota:** El archivo `train.py` utiliza una URI local de MLflow correspondiente al entorno original. Para ejecutar el entrenamiento en otra computadora, es necesario adaptar esa URI a una ruta accesible en el nuevo entorno.

## 8. Configuraciones experimentales

Se realizaron cinco configuraciones distintas, manteniendo constante la semilla aleatoria `42`.

| Corrida | C | max_iter | tol | Accuracy | F1-score |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.0 | 100 | 0.0001 | 0.7353 | 0.7447 |
| 2 | 0.1 | 200 | 0.0001 | 0.7316 | 0.7402 |
| 3 | 10.0 | 200 | 0.0001 | 0.7353 | 0.7447 |
| 4 | 1.0 | 300 | 0.001 | 0.7316 | 0.7402 |
| 5 | 0.5 | 500 | 0.00001 | 0.7353 | 0.7447 |

## 9. Análisis de resultados

Las configuraciones 1, 3 y 5 obtuvieron los mejores resultados, alcanzando:

- Accuracy: 73.53%.
- Precision: 76.09%.
- Recall: 72.92%.
- F1-score: 74.47%.

Las configuraciones 2 y 4 presentaron un desempeño ligeramente inferior.

Los resultados indican que modificar los hiperparámetros no garantiza una mejora automática del modelo.

El hiperparámetro `C`, que controla la intensidad de regularización, es relevante para el comportamiento de la regresión logística. Sin embargo, los experimentos realizados no permiten determinar de manera concluyente cuál hiperparámetro fue el más influyente, debido a que algunas configuraciones modificaron varios parámetros simultáneamente.

## 10. Reproducibilidad

Se ejecutó nuevamente la configuración inicial utilizando:

```bash
python train.py --C 1.0 --max_iter 100 --tol 0.0001 --seed 42
```

Se obtuvieron nuevamente los siguientes resultados:

| Métrica | Resultado |
| --- | ---: |
| Accuracy | 0.7353 |
| Precision | 0.7609 |
| Recall | 0.7292 |
| F1-score | 0.7447 |

Matriz de confusión:

```
[[ 95  33]
 [ 39 105]]
```

La coincidencia de las métricas y la matriz de confusión confirma la reproducibilidad observada dentro del mismo entorno de ejecución.

## 11. Limitaciones y oportunidades de mejora

- Se utilizó una única partición de entrenamiento y prueba.
- No se implementó validación cruzada.
- Algunas configuraciones modificaron más de un hiperparámetro.
- El desempeño se evaluó con un único algoritmo de clasificación.
- La configuración de seguimiento de MLflow utiliza una ruta local específica.
- Para utilizar el modelo con nuevos datos, es necesario aplicar el mismo escalamiento empleado durante el entrenamiento.

Como mejora futura, se recomienda implementar validación cruzada, realizar una búsqueda sistemática de hiperparámetros y registrar el preprocesamiento junto con el modelo.

## 12. Conclusión

Este proyecto permitió implementar un flujo de entrenamiento organizado, parametrizado y reproducible, incorporando buenas prácticas de MLOps mediante MLflow.

Más allá de obtener una métrica de desempeño, el principal aprendizaje fue comprender la importancia de documentar las condiciones de entrenamiento, registrar los experimentos y comparar resultados de manera sistemática.

La trazabilidad y reproducibilidad son fundamentales para desarrollar soluciones de Machine Learning confiables y facilitar su mantenimiento y mejora continua.

---

**Autora:** Berenice Chapuli  
**Repositorio:** [https://github.com/BereniceChapuli/wine-quality-mlops](https://github.com/BereniceChapuli/wine-quality-mlops)
