import argparse
import os
import mlflow
import mlflow.keras
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=str, default="<TRAIN_DATA_PATH>")
    parser.add_argument("--numerical-features", nargs="+", default=["<NUM_FEATURE_1>", "<NUM_FEATURE_2>"])
    parser.add_argument("--categorical-features", nargs="+", default=["<CAT_FEATURE_1>", "<CAT_FEATURE_2>"])
    parser.add_argument("--target", type=str, default="<TARGET_COLUMN>")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--model-dir", type=str, default="model")
    return parser.parse_args()

def build_model(hp, input_dim, num_classes):
    model = keras.Sequential([
        layers.Dense(
            hp.Int("units1", min_value=160, max_value=224, step=32),
            activation="relu",
            input_shape=(input_dim,)
        ),
        layers.BatchNormalization(),
        layers.Dropout(hp.Float("dropout1", min_value=0.25, max_value=0.35, step=0.05)),
        layers.Dense(
            hp.Int("units2", min_value=96, max_value=160, step=32),
            activation="relu"
        ),
        layers.BatchNormalization(),
        layers.Dropout(hp.Float("dropout2", min_value=0.1, max_value=0.3, step=0.05)),
        layers.Dense(num_classes, activation="softmax")
    ])
    lr = hp.Choice("learning_rate", values=[0.0005, 0.0003, 0.0001])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    args = parse_args()
    mlflow.set_experiment("product-mix-experiment")
    with mlflow.start_run() as run:
        df = pd.read_csv(args.train_data)
        all_cols = args.numerical_features + args.categorical_features + [args.target]
        df = df.dropna(subset=all_cols)
        df = shuffle(df, random_state=42)

        for cat_col in args.categorical_features:
            df[cat_col] = df[cat_col].astype(str)
            le = LabelEncoder()
            df[cat_col + "_enc"] = le.fit_transform(df[cat_col])

        cat_enc_features = [c + "_enc" for c in args.categorical_features]
        X_num = df[args.numerical_features].values
        scaler = StandardScaler()
        X_num_scaled = scaler.fit_transform(X_num)
        X_cat = df[cat_enc_features].values
        X = np.concatenate([X_num_scaled, X_cat], axis=1)

        label_y = LabelEncoder()
        y = label_y.fit_transform(df[args.target])
        ros = RandomOverSampler(random_state=42)
        X_res, y_res = ros.fit_resample(X, y)
        x_train, x_test, y_train, y_test = train_test_split(
            X_res, y_res, test_size=0.2, random_state=42
        )

        def tuner_builder(hp):
            return build_model(hp, X.shape[1], len(np.unique(y_res)))

        tuner = kt.Hyperband(
            tuner_builder,
            objective="val_accuracy",
            max_epochs=30,
            factor=3,
            directory="keras_tuner",
            project_name="product_pair_recommendation"
        )

        tuner.search(
            x_train,
            y_train,
            validation_data=(x_test, y_test),
            epochs=30,
            batch_size=32,
            verbose=1
        )

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        mlflow.log_params({
            "units1": best_hps.get("units1"),
            "dropout1": best_hps.get("dropout1"),
            "units2": best_hps.get("units2"),
            "dropout2": best_hps.get("dropout2"),
            "learning_rate": best_hps.get("learning_rate")
        })

        model = tuner.hypermodel.build(best_hps)
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_test, y_test),
            epochs=args.epochs,
            batch_size=32,
            verbose=1
        )

        input_weights = model.layers[0].get_weights()[0]
        importance = np.mean(np.abs(input_weights), axis=1)
        all_feats = args.numerical_features + cat_enc_features
        fi = dict(zip(all_feats, importance))
        fi_sorted = dict(sorted(fi.items(), key=lambda i: i[1], reverse=True))
        for k, v in fi_sorted.items():
            mlflow.log_metric(f"feat_imp_{k}", v)

        yp = model.predict(x_test)

        def precision_at_k(t, s, k=5):
            n = len(t)
            topk = np.argsort(s, axis=1)[:, -k:]
            return sum(t[i] in topk[i] for i in range(n)) / n

        def recall_at_k(t, s, k=5):
            return precision_at_k(t, s, k)

        def average_precision_at_k(t, s, k=5):
            n = len(t)
            si = np.argsort(-s, axis=1)
            ap = 0
            for i in range(n):
                rank = np.where(si[i] == t[i])[0]
                if len(rank) and rank[0] < k:
                    ap += 1/(rank[0]+1)
            return ap / n

        def mean_average_precision_at_k(t, s, k=5):
            return average_precision_at_k(t, s, k)

        for kv in [1,3,5]:
            p_k = precision_at_k(y_test, yp, kv)
            r_k = recall_at_k(y_test, yp, kv)
            map_k = mean_average_precision_at_k(y_test, yp, kv)
            mlflow.log_metric(f"precision@{kv}", p_k)
            mlflow.log_metric(f"recall@{kv}", r_k)
            mlflow.log_metric(f"map@{kv}", map_k)

        pred = np.argmax(yp, axis=1)
        cr = classification_report(y_test, pred, zero_division=1, output_dict=True)
        for lbl, metrics_dict in cr.items():
            if isinstance(metrics_dict, dict):
                for met, val in metrics_dict.items():
                    mlflow.log_metric(f"{lbl}_{met}", val)
        cm = confusion_matrix(y_test, pred)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=False, cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")

        sample_data = x_test[:10]
        sp = model.predict(sample_data)
        st = np.argsort(sp, axis=1)[:, -5:]
        mlflow.keras.log_model(model, artifact_path="model", registered_model_name="product_mix_model_prod")
        model.save(os.path.join(args.model_dir, "model.h5"))

if __name__ == "__main__":
    main()


#mlflow run . \
#-P train_data=mydata.csv \
#-P numerical_features="day_of_week month ticket_price" \
#-P categorical_features="city item_name_location" \
#-P target=item \
#-P epochs=40