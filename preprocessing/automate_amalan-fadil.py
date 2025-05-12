import pandas as pd
from sklearn.preprocessing import LabelEncoder

def automate_preprocessing(input_path: str, output_path: str):
    # Load data
    df = pd.read_csv(input_path)

    # Split Blood Pressure into Systolic and Diastolic
    if 'Blood Pressure' in df.columns:
        df[['Systolic', 'Diastolic']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
        df.drop(columns=['Blood Pressure'], inplace=True)

    # Fill missing values in target column
    if 'Sleep Disorder' in df.columns:
        df['Sleep Disorder'] = df['Sleep Disorder'].fillna('Normal')

    # Drop irrelevant columns
    if 'Person ID' in df.columns:
        df.drop(columns=['Person ID'], inplace=True)

    # Label encode target
    le = LabelEncoder()
    df['Sleep Disorder'] = le.fit_transform(df['Sleep Disorder'])

    # Save cleaned dataset
    df.to_csv(output_path, index=False)
    print(f"✅ Preprocessing selesai. Dataset disimpan di {output_path}")

if __name__ == "__main__":
    input_path = "../Sleep_health_and_lifestyle_dataset.csv"
    output_path = "sleep_data_cleaned.csv"
    automate_preprocessing(input_path, output_path)
