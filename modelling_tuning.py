import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# KONFIGURASI DAGSHUB 
os.environ['MLFLOW_TRACKING_USERNAME'] = 'erzafrian'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '1be7daff1a9fb5b3f0fbdf5aa6cd9d75fe594348'
mlflow.set_tracking_uri('https://dagshub.com/erzafrian/Membangun_Model_Erza-Afrian-Fadillah.mlflow')

def train_and_log():
    mlflow.set_experiment("KNN_Tuning_Final")

    try:
        # Load Dataset
        df = pd.read_csv('customer_behavior_processed.csv')
        X = df.drop('Customer_Rating', axis=1)
        y = df['Customer_Rating']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Hyperparameter Tuning
        param_grid = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
        grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=3)

        with mlflow.start_run(run_name="KNN_Manual_Artifact_Logging"):
            print("🚀 Training model...")
            grid_search.fit(X_train, y_train)
            knn = grid_search.best_estimator_
            y_pred = knn.predict(X_test)

            # Manual Metrics Logging
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred, average='weighted')
            }
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metrics(metrics)

            # INTEGRASI KODE REQUEST ANDA
            signature = infer_signature(X_train, knn.predict(X_train))
            
            # Membuat direktori lokal untuk menyimpan model sementara
            local = os.path.join("artifacts", "knn_model")
            if os.path.exists(local):
                shutil.rmtree(local) 
            mlflow.sklearn.save_model(
                knn, local, input_example=X_train[:5], signature=signature
            )

            # Log setiap file dari direktori lokal ke MLflow artifact path "knn_model"
            print("Mengunggah artefak model dari folder lokal...")
            for file in os.listdir(local):
                mlflow.log_artifact(os.path.join(local, file), artifact_path="knn_model")

            # Confusion Matrix 
            plt.figure(figsize=(8,6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
            plt.title('Confusion Matrix')
            plot_path = "confusion_matrix.png"
            plt.savefig(plot_path)
            mlflow.log_artifact(plot_path, artifact_path="knn_model")
            plt.close()

            # Simpan Classification Report ke dalam folder 'knn_model'
            report_path = "classification_report.txt"
            with open(report_path, "w") as f:
                f.write(classification_report(y_test, y_pred))
            mlflow.log_artifact(report_path, artifact_path="knn_model")

            print(f"Selesai!")

    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

if __name__ == "__main__":
    train_and_log()
