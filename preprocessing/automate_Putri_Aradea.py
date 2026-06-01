import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_data(input_path, output_path):
    """
    Fungsi untuk melakukan preprocessing dataset stroke
    """

    #load dataset
    df = pd.read_csv(input_path)

    #menangani missing value
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())

    #menghapus data duplikat
    df = df.drop_duplicates()

    #menghapus kolom yg tdk perlu
    df = df.drop(columns=['id'])

    #encoding data kategori
    df = pd.get_dummies(
        df,
        columns=[
            'gender',
            'ever_married',
            'work_type',
            'Residence_type',
            'smoking_status'
        ],
        drop_first=True
    )

    #standarisasi fitur numerik
    scaler = StandardScaler()

    numerical_cols = [
        'age',
        'avg_glucose_level',
        'bmi'
    ]

    df[numerical_cols] = scaler.fit_transform(
        df[numerical_cols]
    )

    #simpan hasil
    df.to_csv(output_path, index=False)

    print(f"Preprocessing selesai!")
    print(f"Dataset tersimpan di: {output_path}")


if __name__ == "__main__":
    preprocess_data(
        input_path="src/Eksperimen_SML_Putri_Aradea/healthcare-dataset-stroke-data.csv",
        output_path="src/Eksperimen_SML_Putri_Aradea/preprocessing/stroke_preprocessed.csv"
    )