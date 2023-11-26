import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats
import librosa
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.decomposition import PCA as sklearnPCA
from joblib import dump

st.title("PROYEK SAINS DATA")
st.write("Nama  : Jennatul Macwe ")
st.write("Nim   : 210411100151 ")
st.write("Kelas : Proyek Sains Data A ")

data_set_description, pre, knn, upload_data = st.tabs(["Deskripsi Data Set", "Pre-prosessing", "K-NN Model", "Upload Data"])

with data_set_description:
    st.write("### Deskripsi Data Set")
    st.write("Dataset audio ini terdapat 200 kata target yang diucapkan dalam frasa oleh dua aktris (berusia 26 dan 64 tahun). Rekaman dibuat dari set tersebut yang menggambarkan tujuh emosi (marah, jijik, takut, bahagia, kejutan, menyenangkan, kesedian dan netral). Dataset ini terdiri dari 2746 data dengan 14 kelas dimana dua aktor merekan masing-masing 7 emosi. ") 
    st.write("Data set ini adalah data set sinyal audio yang telah dilakukan perhitungan statistika. Di mana data yang digunakan sebanyak 2810. Terdapat 10 fitur dalam perhitungan data ini, diantaranya yaitu ZCR Mean, ZCR Median, ZCR Standar Deviasi, ZCR Kurtosis, ZCR Skewness, RMSE, RMSE Median, RMSE Standar Deviasi, RMSE Kurtosis, dan RMSE Skewness.")
    st.write("### Fitur")
    st.write("Pada dataset ini menjadikan perhitungan statistika sebagai fitur, dimana fitur-fitur tersebut terdiri sebanyak 10 fitur yaitu:")
    st.write("* zcr_mean")
    st.write("* zcr_median")
    st.write("* zcr_std_dev")
    st.write("* zcr_kurtosis")
    st.write("* zcr_skew")
    st.write("* rmse")
    st.write("* rmse_median")
    st.write("* rmse_std_dev")
    st.write("* rmse_kurtosis")
    st.write("* rmse_skew")
    st.write("### Sumber Data Set Kaggle ")
    st.write("Dataset Audio ini dapat dilihat di kaggle melalui link di bawah ini:")
    st.write("https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess?resource=download")
    
with pre:
    st.write("### Data Sinyal Audio")
    df = pd.read_csv('https://raw.githubusercontent.com/jennamacwe/ProyekSainData/main/datEmosi.csv')
    st.dataframe(df)
    
    # Baca data dari file CSV
    dataknn = df

    # Pisahkan fitur (X) dan label (y)
    X = dataknn.drop(['Label'], axis=1)
    y = dataknn['Label']

    # Split data menjadi set pelatihan dan pengujian
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)

    # Tentukan scaler
    scaler = StandardScaler()

    # Fit scaler pada dataset pelatihan
    scaler.fit(X_train)

    # Simpan scaler
    dump(scaler, open('scaler.pkl', 'wb'))

    # Transformasikan dataset pelatihan
    X_train_scaled = scaler.transform(X_train)

    # Menambahkan widget Streamlit untuk menampilkan dataset pelatihan
    # st.title('')
    st.write("### Data Train yang telah di normalisasi")
    st.dataframe(X_train)

    # Menambahkan widget Streamlit untuk menampilkan dataset pengujian
    st.write("### Data Test yang telah di normalisasi")
    st.dataframe(X_test)

    # Menambahkan widget Streamlit untuk menampilkan informasi tentang scaler
    # st.title('')
    st.write("### Scaler Information")
    st.text("Scaler Mean: {}".format(scaler.mean_))
    st.text("Scaler Scale: {}".format(scaler.scale_))

    # setelah menganalisis audio, Anda dapat melakukan reduksi data dengan random sampling
    st.write("### Reduksi Data dengan Random Sampling")
    reduced_df = df.sample(frac=0.5, random_state=42)  # contoh mengambil 50% sampel secara acak

    # tampilkan hasil data yang sudah direduksi
    st.write("Data setelah di reduksi.")
    st.dataframe(reduced_df)

with knn:
    st.write("### Data Sinyal Audio")
    st.write("Ini adalah kumpulan data sinyal audio emosi yang mencakup perhitungan berbagai statistik, termasuk ZCR Mean, ZCR Median, ZCR Standar Deviasi, ZCR Kurtosis, ZCR Skewness, RMSE, RMSE Median, RMSE Standar Deviasi, RMSE Kurtosis, dan RMSE Skewness.")
    df = pd.read_csv('https://raw.githubusercontent.com/jennamacwe/ProyekSainData/main/datEmosi.csv')
    st.dataframe(df)

    # Memisahkan fitur (X) dan label (y)
    X = df.drop(['Label'], axis=1)
    y = df['Label']

    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)

    # Define and fit the scaler on the training dataset/normalisasi data
    scaler = StandardScaler()
    scaler.fit(X_train)

    # Save the scaler using pickle/hasil normalisasi disimpan dalam scaler
    scaler_file_path = r'scaler.pkl'
    with open(scaler_file_path, 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

    X_train_scaled = scaler.transform(X_train)

    with open(r'scaler.pkl', 'rb') as normalisasi:
        loadscal = pickle.load(normalisasi)

    X_test_scaled = loadscal.transform(X_test)

    # Hitung akurasi KNN dari k = 1 hingga 30
    K = 30
    acc = np.zeros((K - 1))

    for n in range(1, K, 2):
        knn = KNeighborsClassifier(n_neighbors=n, metric="euclidean").fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)
        acc[n - 1] = accuracy_score(y_test, y_pred)

    best_accuracy = acc.max()
    best_k = acc.argmax() + 1

    # Tampilkan akurasi terbaik dan nilai k
    st.write('Akurasi KNN terbaik adalah', best_accuracy, 'dengan nilai k =', best_k)

    # Simpan model KNN terbaik
    best_knn = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean")
    best_knn.fit(X_train_scaled, y_train)

    # Save the best KNN model using pickle
    model_file_path = r'KNNmodel.pkl'
    with open(model_file_path, 'wb') as model_file:
        pickle.dump(best_knn, model_file)

    with open(r'KNNmodel.pkl', 'rb') as knn_model:
        load_knn = pickle.load(knn_model)
    y_pred = load_knn.predict(X_test_scaled)

    # Hitung dan tampilkan akurasi KNN
    st.write('Akurasi KNN dengan data test:')
    accuracy = accuracy_score(y_test, y_pred)
    st.write(accuracy)

    # Hitung prediksi label KNN
    knn_predictions = load_knn.predict(X_test_scaled)

    # Simpan hasil prediksi KNN ke dalam DataFrame
    knn_results_df = pd.DataFrame({'Actual Label': y_test, 'Predicted Label (KNN)': knn_predictions})

    # Tampilkan tabel prediksi KNN
    st.write("### Tabel Prediksi Label KNN")
    st.dataframe(knn_results_df)

    # Lakukan reduksi PCA
    sklearn_pca = sklearnPCA(n_components=8)
    X_train_pca = sklearn_pca.fit_transform(X_train_scaled)
    st.write("Principal Components 8:")
    st.write(X_train_pca)

    # Save the PCA model
    pca_model_file_path = r'PCA8.pkl'
    with open(pca_model_file_path, 'wb') as pca_model_file:
        pickle.dump(sklearn_pca, pca_model_file)

    # Load the PCA model
    with open(pca_model_file_path, 'rb') as pca_model:
        loadpca = pickle.load(pca_model)

    # Transform test data using the loaded PCA model
    X_test_pca = loadpca.transform(X_test_scaled)

    # Continue with KNN and evaluation as needed/data yang sudah direduksi dihtung KNN
    K = 30
    acc_pca = np.zeros((K - 1))
    for n in range(1, K, 2):
        knn_pca = KNeighborsClassifier(n_neighbors=n, metric="euclidean").fit(X_train_pca, y_train)
        y_pred_pca = knn_pca.predict(X_test_pca)
        acc_pca[n - 1] = accuracy_score(y_test, y_pred_pca)

    best_accuracy_pca = acc_pca.max()
    best_k_pca = acc_pca.argmax() + 1

    # Tampilkan akurasi terbaik dan nilai k dengan PCA
    st.write('Akurasi KNN terbaik dengan PCA adalah', best_accuracy_pca, 'dengan nilai k =', best_k_pca)

    # Hitung prediksi label KNN setelah PCA
    knn_pca_predictions = knn_pca.predict(X_test_pca)

    # Simpan hasil prediksi KNN setelah PCA ke dalam DataFrame
    knn_pca_results_df = pd.DataFrame({'Actual Label': y_test, 'Predicted Label (KNN with PCA)': knn_pca_predictions})

    # Tampilkan tabel prediksi KNN setelah PCA
    st.write("### Tabel Prediksi Label KNN dengan PCA")
    st.dataframe(knn_pca_results_df)

def extract_features(audio_path):
    y, sr = librosa.load(audio_path)

    # Fitur 1: Mean
    mean = np.mean(y)

    # Fitur 2: Median
    median = np.median(y)

    # Fitur 3: Standard Deviation
    std_deviation = np.std(y)

    # Fitur 4: Zero Crossing Rate
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))

    # Fitur 5: Energy
    energy = np.mean(librosa.feature.rms(y=y))

    # Fitur 6: Spectral Centroid
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    # Fitur 7: Spectral Bandwidth
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

    # Fitur 8: Spectral Roll-off
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

    # Fitur 9: Chroma Feature
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))

    # Fitur 10: Mel-frequency Cepstral Coefficients (MFCCs)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13))

    return [mean, median, std_deviation, zero_crossing_rate, energy,
            spectral_centroid, spectral_bandwidth, spectral_rolloff, chroma, mfccs]

with upload_data:
    def main():
        st.title('Ekstraksi Fitur Audio')
        st.write('Unggah file audio WAV untuk menghitung fitur statistiknya.')

        # Unggah file audio
        uploaded_audio = st.file_uploader("Pilih file audio", type=["wav"])

        if uploaded_audio is not None:
            st.audio(uploaded_audio, format='audio/wav', start_time=0)

            audio_features = extract_features(uploaded_audio)
            audio_features_reshaped = np.array(audio_features).reshape(1, -1)  # Reshape to 2D array

            feature_names = [
                "Mean", "Median", "Std Deviation", "Zero Crossing Rate", "Energy",
                "Spectral Centroid", "Spectral Bandwidth", "Spectral Rolloff", "Chroma", "MFCCs"
            ]

            # Tampilkan hasil fitur
            st.write("### Hasil Ekstraksi Fitur Audio:")
            for i, feature in enumerate(audio_features):
                st.write(f"{feature_names[i]}: {feature}")

            # Transform audio_features using the loaded scaler
            datauji = loadscal.transform(audio_features_reshaped)
            datapca = loadpca.transform(datauji)

            # Make predictions using the KNN model
            y_pred_uji = load_knn.predict(datauji)

            st.write("Fitur yang sudah di normalisasi: ", datauji)
            st.write("Data PCA:", datapca)
            st.write("Prediksi Label menggunakan KNN:", y_pred_uji)

    if __name__ == "__main__":
        main()