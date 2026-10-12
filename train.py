import pandas as pd
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,confusion_matrix)


def cargar_datos(ruta):
	"""
	Carga el dataset Wine Quality y elimina registros duplicados.
	"""
	wine_df = pd.read_csv(ruta, sep=";")

	print("Registros originales:", len(wine_df))

	wine_df = wine_df.drop_duplicates().reset_index(drop=True)

	print("Registros después de limpieza:", len(wine_df))

	return wine_df
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preparar_datos(wine_df, seed=42):
    """
    Prepara las variables, divide los datos y aplica escalamiento.
    """

    # Crear variable objetivo binaria
    wine_df = wine_df.copy()
    wine_df["quality_class"] = (wine_df["quality"] >= 6).astype(int)

    # Separar variables predictoras y objetivo
    X = wine_df.drop(columns=["quality", "quality_class"])
    y = wine_df["quality_class"]

    # Dividir datos: 80% entrenamiento y 20% prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=seed,
        stratify=y
    )

    # Escalar datos sin utilizar información del conjunto de prueba
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
    from sklearn.linear_model import LogisticRegression


def entrenar_modelo(X_train, y_train, C=1.0, max_iter=100, tol=0.0001, seed=42):
    """
    Entrena un modelo de regresión logística con
    hiperparámetros configurables.
    """

    modelo = LogisticRegression(
        C=C,
        max_iter=max_iter,
        tol=tol,
        random_state=seed
    )

    modelo.fit(X_train, y_train)

    return modelo

    from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def evaluar_modelo(modelo, X_test, y_test):
    """
    Evalúa el modelo y devuelve sus métricas
    y la matriz de confusión.
    """

    # Realizar predicciones
    y_pred = modelo.predict(X_test)

    # Calcular métricas
    metricas = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0)
    }

    # Generar matriz de confusión
    matriz = confusion_matrix(y_test, y_pred, labels=[0, 1])

    # Mostrar resultados
    print("\nResultados del modelo:")
    for nombre, valor in metricas.items():
        print(f"{nombre}: {valor:.4f}")

    print("\nMatriz de confusión:")
    print(matriz)

    return metricas, matriz

    import argparse


def obtener_argumentos():
    """
    Configura los argumentos para ejecutar el modelo
    desde la terminal.
    """

    parser = argparse.ArgumentParser(
        description="Entrenamiento del modelo Wine Quality"
    )

    parser.add_argument(
        "--data",
        type=str,
        default="data/winequality-red.csv"
    )

    parser.add_argument(
        "--C",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--max_iter",
        type=int,
        default=100
    )

    parser.add_argument(
        "--tol",
        type=float,
        default=0.0001
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    return parser.parse_args()

    import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from pathlib import Path


def registrar_experimento(modelo, metricas, matriz, args):
    """
    Registra parámetros, métricas, matriz de confusión
    y modelo entrenado en MLflow.
    """

# Configurar MLflow con una base de datos local del proyecto
mlflow.set_tracking_uri("sqlite:///mlflow.db")

    # Usar un experimento nuevo para esta actividad
    mlflow.set_experiment("Wine_Quality_Hyperparameters")

    with mlflow.start_run():

        # Registrar parámetros
        mlflow.log_param("C", args.C)
        mlflow.log_param("max_iter", args.max_iter)
        mlflow.log_param("tol", args.tol)
        mlflow.log_param("seed", args.seed)
        mlflow.log_param("test_size", 0.20)
        mlflow.log_param("escalamiento", "StandardScaler")

        # Registrar métricas
        for nombre, valor in metricas.items():
            mlflow.log_metric(nombre, valor)

        # Crear matriz de confusión
        fig, ax = plt.subplots(figsize=(6, 5))

        ConfusionMatrixDisplay(
            confusion_matrix=matriz,
            display_labels=["Calidad baja", "Calidad alta"]
        ).plot(ax=ax, cmap="Blues")

        plt.title("Matriz de confusión")
        fig.tight_layout()

        # Guardar imagen como artefacto
        Path("artifacts").mkdir(exist_ok=True)
        ruta_imagen = "artifacts/matriz_confusion.png"

        fig.savefig(ruta_imagen)
        plt.close(fig)

        mlflow.log_artifact(ruta_imagen)

        # Registrar modelo entrenado
        mlflow.sklearn.log_model(
            sk_model=modelo,
            name="modelo_regresion_logistica"
        )

        print("\nExperimento registrado correctamente en MLflow.")
def main():
    """
    Ejecuta el proceso completo de entrenamiento.
    """

    # 1. Obtener los argumentos de la terminal
    args = obtener_argumentos()

    # 2. Cargar y limpiar los datos
    wine_df = cargar_datos(args.data)

    # 3. Preparar los datos
    X_train, X_test, y_train, y_test, scaler = preparar_datos(
        wine_df,
        seed=args.seed
    )

    # 4. Entrenar el modelo
    modelo = entrenar_modelo(
        X_train,
        y_train,
        C=args.C,
        max_iter=args.max_iter,
        tol=args.tol,
        seed=args.seed
    )

    # 5. Evaluar el modelo
    metricas, matriz = evaluar_modelo(
        modelo,
        X_test,
        y_test
    )

    # 6. Registrar los resultados en MLflow
    registrar_experimento(
        modelo,
        metricas,
        matriz,
        args
    )


if __name__ == "__main__":
    main()