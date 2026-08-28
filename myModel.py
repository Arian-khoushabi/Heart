# 1.Import libraries........
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.compose import ColumnTransformer
import joblib
from model_preprocess import HeartModel, CustomTransformer

if __name__ == "__main__":

    # 2.Reading the dataset........
    data = pd.read_csv("UCI_Heart_Disease_Dataset_Combined.csv")
    columns_to_object = ["Sex", "ChestPainType", "FastingBS", "RestingECG", "ExerciseAngina", "HeartDisease"]
    for col in columns_to_object:
        data[col] = data[col].astype("category")
    cat_col = data.select_dtypes(include="category").columns.to_list()
    cat_col.remove("HeartDisease")
    num_col = data.select_dtypes(exclude="category").columns.to_list()

    # 3.Handling Null values........
    data.loc[data["Cholesterol"] == 0, "Cholesterol"] = data.loc[data["Cholesterol"] != 0, "Cholesterol"].median()
    data.loc[data["RestingBP"] == 0, "RestingBP"] = data.loc[data["RestingBP"] != 0, "RestingBP"].median()

    # 4.Spliting data........
    X = data.drop("HeartDisease", axis=1)
    y = data["HeartDisease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    # 5.Using transformers........
    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    pt.fit(X_train[["Oldpeak"]])
    transformed = pt.transform(X_train[["Oldpeak"]])
    down_clip = np.quantile(transformed, 0.01)
    up_clip = np.quantile(transformed, 0.99)
    custom_transformer = CustomTransformer(pt, down_clip, up_clip)

    # 6.Preprocessing........
    column_transfomer = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_col)
    ], remainder="passthrough")

    # 7.Building the model........
    pipeline = Pipeline([
        ("custom", custom_transformer),
        ("preprocess", column_transfomer),
        ("model", XGBClassifier(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=6,
            min_child_weight=8,
            subsample=1.0,
            colsample_bytree=1.0,
            random_state=42,
            reg_alpha=1.2,
            reg_lambda=1.4,
            n_jobs=-1,
            scale_pos_weight=1.10,
            eval_metric="logloss"
        ))
    ])
    pipeline.fit(X_train, y_train)

    # 8.Test scores........
    def best_predict(model, X_test, y_test):
        y_proba = model.predict_proba(X_test)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
        beta = 2
        precision_adj = precisions[:-1]
        recall_adj = recalls[:-1]
        f2_scores = (1 + beta**2) * (precision_adj * recall_adj) / ((beta**2 * precision_adj) + recall_adj)
        best_idx = np.argmax(f2_scores)
        best_threshold = thresholds[best_idx]
        y_predicts = (y_proba >= best_threshold).astype(int)
        return y_predicts, best_threshold

    y_predicts, best_threshold = best_predict(pipeline, X_test, y_test)

    # 9.Deploy the model........
    final_model = HeartModel(
        pipeline=pipeline,
        threshold=best_threshold,
        columns=X_train.columns.tolist(),
        cat_columns=cat_col
    )
    joblib.dump(final_model, "Heart_ai.pkl")
    print("model saved successfully")