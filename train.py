import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np


def train_and_save_model(csv_file='Training.csv', target_column='disease'):
    try:
        data = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"ERROR: The file '{csv_file}' was not found.")
        return None, None

    if target_column not in data.columns:
        print(f"ERROR: Target column '{target_column}' not found.")
        return None, None

    data.dropna(subset=[target_column], inplace=True)

    # 1. Encode the ENTIRE disease column.
    label_encoder = LabelEncoder()
    data[target_column] = label_encoder.fit_transform(data[target_column])

    # 2. Find single-sample classes (AFTER encoding).
    label_counts = data[target_column].value_counts()
    single_sample_labels = label_counts[label_counts == 1].index.tolist()

    # 3. Handle "Other" class.
    other_label = len(label_encoder.classes_)  # The NEXT integer.
    if single_sample_labels:
        print(f"Combining the following single-sample disease(s) into 'Other':")
        for label in single_sample_labels:
            # Inverse transform to show the ORIGINAL names.
            original_name = label_encoder.inverse_transform([label])[0]
            print(f"  - {original_name} (encoded as {label})")

        # Combine into "Other" (represented by other_label).
        data[target_column] = data[target_column].apply(
            lambda x: other_label if x in single_sample_labels else x
        )
    symptom_cols = [col for col in data.columns if col != target_column]
    X = pd.DataFrame(0, index=np.arange(len(data)), columns=symptom_cols)
    for col in symptom_cols:
        X[col] = data[col].apply(lambda x: 0 if pd.isna(x) else 1)

    y_encoded = data[target_column]


    # --- MODEL TRAINING ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # --- SAVE DATA ---
    symptoms = list(X.columns)  # The symptom names are the column names of X.
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    with open('symptoms.pkl', 'wb') as f:
        pickle.dump(symptoms, f)


    return accuracy
if __name__ == '__main__':
    accuracy = train_and_save_model()
    if accuracy is not None:
        print(f"Accuracy: {accuracy}")

