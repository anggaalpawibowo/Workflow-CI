import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Mengaktifkan fitur Autologging (Syarat Wajib Kriteria 2)
mlflow.sklearn.autolog()

def main():
    # 2. Menentukan Tracking URI (Lokal) dan Nama Eksperimen
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Eksperimen_SML_Angga")

    with mlflow.start_run(run_name="Basic_RF_Model"):
        # 3. Memuat Data
        print("Memuat dataset...")
        df = pd.read_csv('dataset_preprocessing/crop_data_clean.csv')
        
        # Dataset crop_data memiliki target di kolom 'label'
        X = df.drop(columns=['label'])
        y = df['label']
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 4. Inisialisasi dan Melatih Model (Tanpa Hyperparameter Tuning)
        print("Melatih model Random Forest...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # 5. Prediksi dan Cetak Akurasi (Metrik dan artefak sudah dicatat otomatis oleh autolog)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Proses selesai. Akurasi Model: {acc:.4f}")

if __name__ == "__main__":
    main()