# 1.Import libraries........
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score
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

    # 3.Spliting data........
    X = data.drop("HeartDisease", axis=1)
    y = data["HeartDisease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,

        random_state=42
    )
    X_fit, X_val, y_fit, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    stratify=y_train,
    random_state=42
    )


    # 4.Handling Null values........
    X_fit = X_fit.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    cholesterol_median = X_fit.loc[
        X_fit["Cholesterol"] != 0,
        "Cholesterol"
    ].median()

    restingbp_median = X_fit.loc[
        X_fit["RestingBP"] != 0,
        "RestingBP"
    ].median()

    X_fit.loc[
        X_fit["Cholesterol"] == 0,
        "Cholesterol"
    ] = cholesterol_median

    X_val.loc[
        X_val["Cholesterol"] == 0,
        "Cholesterol"
    ] = cholesterol_median

    X_test.loc[
        X_test["Cholesterol"] == 0,
        "Cholesterol"
    ] = cholesterol_median

    X_fit.loc[
        X_fit["RestingBP"] == 0,
        "RestingBP"
    ] = restingbp_median

    X_val.loc[
        X_val["RestingBP"] == 0,
        "RestingBP"
    ] = restingbp_median

    X_test.loc[
        X_test["RestingBP"] == 0,
        "RestingBP"
    ] = restingbp_median

    # 5.Using transformers........
    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    pt.fit(X_fit[["Oldpeak"]])
    transformed = pt.transform(X_fit[["Oldpeak"]])
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
    pipeline.fit(X_fit, y_fit)

    # 8.Test scores........
    def find_best_threshold(model, X_val, y_val, min_recall=0.90):

        y_proba = model.predict_proba(X_val)[:, 1]

        best_threshold = None
        best_specificity = -1
        best_recall = 0
        best_precision = 0

        for threshold in np.linspace(0.01, 0.99, 990):

            y_pred = (
                y_proba >= threshold
            ).astype(int)

            tn, fp, fn, tp = confusion_matrix(
                y_val,
                y_pred
            ).ravel()

            recall = tp / (tp + fn)

            specificity = tn / (tn + fp)

            precision = (
                tp / (tp + fp)
                if (tp + fp) > 0
                else 0
            )

            if recall >= min_recall:

                if specificity > best_specificity:

                    best_threshold = threshold
                    best_specificity = specificity
                    best_recall = recall
                    best_precision = precision

        print(f"Minimum required recall: {min_recall:.4f}")
        print(f"Selected threshold: {best_threshold:.4f}")
        print(f"Validation recall: {best_recall:.4f}")
        print(f"Validation specificity: {best_specificity:.4f}")
        print(f"Validation precision: {best_precision:.4f}")

        return best_threshold
    
    best_threshold = find_best_threshold(
    pipeline,
    X_val,
    y_val,
    min_recall=0.85
)

    pipeline.fit(X_train, y_train)
    y_proba_test = pipeline.predict_proba(X_test)[:, 1]

    y_pred_test = (
        y_proba_test >= best_threshold
    ).astype(int)

    print(
    f"accuracy is "
    f"{accuracy_score(y_test, y_pred_test):.4f}"
    )
    print(
        f"precision is "
        f"{precision_score(y_test, y_pred_test):.4f}"
    )

    print(
        f"recall is "
        f"{recall_score(y_test, y_pred_test):.4f}"
    )

    print(
        f"f1 is "
        f"{f1_score(y_test, y_pred_test):.4f}"
    )

    print(
        f"roc-auc is "
        f"{roc_auc_score(y_test, y_proba_test):.4f}"
    )

    print(
        f"pr-auc is "
        f"{average_precision_score(y_test, y_proba_test):.4f}"
    )
    
    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred_test
    ).ravel()
    print(
        f"Specificity is "
        f"{tn / (tn + fp):.4f}"
    )


    # 9.Deploy the model........
    final_model = HeartModel(
        pipeline=pipeline,
        threshold=best_threshold,
        columns=X_train.columns.tolist(),
        cat_columns=cat_col
    )
    joblib.dump(final_model, "Heart_ai.pkl")
    print("model saved successfully")