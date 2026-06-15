from flask import Flask, render_template, request
import pickle
import pandas as pd
import traceback

app = Flask(__name__)

# Memuat model dan scaler yang sudah ditraining
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
        
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
except Exception as e:
    print(f"Gagal memuat model.pkl atau scaler.pkl: {e}")

# Mendapatkan nama model
# Jika model.pkl dari train.py (bentuknya dictionary/kamus)
if isinstance(model, dict):
    model_names = list(model.keys())
else:
    # Jika model.pkl dari versi bawaan awal (bentuknya list)
    model_names = ['Decision Tree', 'SVC']

@app.route('/')
def index():
    return render_template('index.html', model_names=model_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        selected_model_name = request.form['model']
        
        # Mengambil data dari form HTML dan memaksanya menjadi angka
        data = {
            'Pregnancies': int(request.form['pregnancies']),
            'Glucose': int(request.form['glucose']),
            'BloodPressure': int(request.form['blood_pressure']),
            'SkinThickness': int(request.form['skin_thickness']),
            'Insulin': int(request.form['insulin']),
            'BMI': float(request.form['bmi']),
            'DiabetesPedigreeFunction': float(request.form['diabetes_pedigree']),
            'Age': int(request.form['age'])
        }
        
        # Mengubah data input menjadi DataFrame
        input_data = pd.DataFrame(data, index=[0])
        
        # Melakukan scaling (standarisasi) pada input
        input_data_scaled = scaler.transform(input_data)
        
        # MENDAPATKAN MODEL YANG TEPAT (Perbaikan Error 500 Utama)
        if isinstance(model, dict):
            selected_model_obj = model[selected_model_name]
        else:
            selected_model_idx = model_names.index(selected_model_name)
            selected_model_obj = model[selected_model_idx]
        
        # Melakukan prediksi
        prediction_result = selected_model_obj.predict(input_data_scaled)
        
        # Menentukan hasil teks
        if prediction_result[0] == 1:
            prediction_text = 'Diabetic (Positif Diabetes)'
        else:
            prediction_text = 'Non-Diabetic (Negatif Diabetes)'
            
        return render_template("index.html", model_names=model_names, prediction=prediction_text)

    except Exception as e:
        # Menangkap error agar web tidak crash (Layar Putih 500)
        print("\n=== TERJADI ERROR SAAT PREDIKSI ===")
        traceback.print_exc() # Menampilkan detail error merah di terminal
        print("=====================================\n")
        
        error_msg = f"Error: Terjadi kesalahan. Pastikan semua kolom terisi dengan benar (tidak boleh kosong)."
        return render_template("index.html", model_names=model_names, prediction=error_msg)

if __name__ == '__main__':
    # Debug=True mempermudah mencari error
    app.run(debug=True)