import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# KONFIGURASI DAGSHUB 
os.environ['MLFLOW_TRACKING_USERNAME'] = 'erzafrian'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '1be7daff1a9fb5b3f0fbdf5aa6cd9d75fe594348'
mlflow.set_tracking_uri('https://dagshub.com/erzafrian/Membangun_Model_Erza-Afrian-Fadillah.mlflow')

def train_and_log_to_dagshub():
    # Menentukan Nama Eksperimen
    mlflow.set_experiment("KNN_Tuning_DagsHub_Erza")

    try:
        # Load Dataset Hasil Preprocessing
        df = pd.read_csv('customer_behavior_preprocessing.csv')
        X = df.drop('Customer_Rating', axis=1)
        y = df['Customer_Rating']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Setup Hyperparameter Tuning (GridSearchCV)
        param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
        
        grid_search = GridSearchCV(
            KNeighborsClassifier(), 
            param_grid, 
            cv=5, 
            scoring='accuracy', 
            n_jobs=-1
        )

        # Memulai MLflow Run
        with mlflow.start_run(run_name="KNN_Manual_Final_Run"):
            print("🚀 Melakukan Hyperparameter Tuning...")
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)

            # MANUAL LOGGING PARAMETERS
            print("Logging Parameters...")
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_param("tuning_method", "GridSearchCV")

            # MANUAL LOGGING METRICS 
            print("Logging Metrics...")
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average='weighted'),
                "recall": recall_score(y_test, y_pred, average='weighted'),
                "f1_score": f1_score(y_test, y_pred, average='weighted')
            }
            mlflow.log_metrics(metrics)

            # Confusion Matrix 
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
            plt.title('Confusion Matrix - KNN Tuned')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            
            plot_file = "confusion_matrix.png"
            plt.savefig(plot_file)
            mlflow.log_artifact(plot_file, artifact_path="models") 
            plt.close()

            # Classification Report-
            report = classification_report(y_test, y_pred)
            report_file = "classification_report.txt"
            with open(report_file, "w") as f:
                f.write(report)
            mlflow.log_artifact(report_file, artifact_path="models") 

            # LOG MODEL 
            mlflow.sklearn.log_model(best_model, "models")

            print(f"Berhasil! Silakan cek DagsHub untuk melihat folder 'models' berisi artefak.")
            print(f"Best Accuracy: {metrics['accuracy']:.4f}")

    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

if __name__ == "__main__":
    train_and_log_to_dagshub()