import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

mlflow.set_tracking_uri("http://127.0.0.1:5000/")

def train_with_tuning():
    # Konfigurasi MLflow
    mlflow.set_experiment("Eksperimen_SML_KNN_Tuning")

    try:
        # Load dataset
        df = pd.read_csv('customer_behavior_processed.csv')
        X = df.drop('Customer_Rating', axis=1)
        y = df['Customer_Rating']

        # Split data untuk evaluasi
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Definisikan Ruang Hyperparameter
        param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }

        # Setup GridSearchCV
        grid_search = GridSearchCV(
            KNeighborsClassifier(),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )

        # Jalankan satu Parent Run di MLflow
        with mlflow.start_run(run_name="KNN_Hyperparameter_Tuning"):
            print("🚀 Melakukan Hyperparameter Tuning (Grid Search)...")
            grid_search.fit(X_train, y_train)

            # Ambil model terbaik
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)

            # Hitung metrik secara manual
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted')
            rec = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            # Manual Logging ke MLflow
            print("Mencatat parameter dan metrik ke MLflow...")
            
            # Log Parameters (Manual)
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_param("tuning_method", "GridSearchCV")
            mlflow.log_param("cv_folds", 5)

            # Log Metrics (Manual)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)

            # Log Model (Manual)
            mlflow.sklearn.log_model(best_model, "knn_tuned_model")

            print(f"Selesai! Best Params: {grid_search.best_params_}")
            print(f"Accuracy Terbaik: {acc:.4f}")

    except FileNotFoundError:
        print("Error: File 'customer_behavior_processed.csv' tidak ditemukan.")

if __name__ == "__main__":
    train_with_tuning()