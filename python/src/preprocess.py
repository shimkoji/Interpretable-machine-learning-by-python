import re

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


def preprocess_bike_data(df_bike: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the bike sharing dataset."""

    df_bike["weekday"] = pd.Categorical(
        df_bike["weekday"], categories=range(7), ordered=True
    ).rename_categories(["SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT"])
    df_bike["holiday"] = pd.Categorical(
        df_bike["holiday"], categories=[0, 1], ordered=True
    ).rename_categories(["NO HOLIDAY", "HOLIDAY"])
    df_bike["workingday"] = pd.Categorical(
        df_bike["workingday"], categories=[0, 1], ordered=True
    ).rename_categories(["NO WORKING DAY", "WORKING DAY"])
    df_bike["season"] = pd.Categorical(
        df_bike["season"], categories=range(1, 5), ordered=True
    ).rename_categories(["WINTER", "SPRING", "SUMMER", "FALL"])
    df_bike["weathersit"] = pd.Categorical(
        df_bike["weathersit"], categories=range(1, 4), ordered=True
    ).rename_categories(["GOOD", "MISTY", "RAIN/SNOW/STORM"])
    df_bike["mnth"] = pd.Categorical(
        df_bike["mnth"], categories=range(1, 13), ordered=True
    ).rename_categories(
        [
            "JAN",
            "FEB",
            "MAR",
            "APR",
            "MAY",
            "JUN",
            "JUL",
            "AUG",
            "SEP",
            "OCT",
            "NOV",
            "DEC",
        ]
    )

    df_bike["yr"] = np.where(df_bike["yr"] == 0, 2011, 2012)
    df_bike["yr"] = pd.Categorical(df_bike["yr"])
    df_bike["dteday"] = pd.to_datetime(df_bike["dteday"])
    df_bike["days_since_2011"] = (df_bike["dteday"] - df_bike["dteday"].min()).dt.days
    df_bike["temp"] = df_bike["temp"] * (39 - (-8)) + (-8)
    df_bike["atemp"] = df_bike["atemp"] * (50 - (16)) + (16)
    df_bike["windspeed"] = 67 * df_bike["windspeed"]
    df_bike["hum"] = 100 * df_bike["hum"]

    return df_bike.drop(columns=["instant", "dteday", "registered", "casual", "atemp"])


def preprocess_ycomments(df_ycomments: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the YouTube Spam Comments dataset."""

    def _clean_ycomments(html_string: str) -> str:
        if pd.isna(html_string):
            return html_string
        return re.sub("<.*?>", "", html_string)

    df_ycomments["CONTENT"] = df_ycomments["CONTENT"].apply(_clean_ycomments)
    # Convert to ASCII
    df_ycomments["CONTENT"] = (
        df_ycomments["CONTENT"]
        .astype(str)
        .str.encode("ascii", "ignore")
        .str.decode("ascii")
    )
    return df_ycomments


def preprocess_rfcc(df_rfcc: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the Risk Factors for Cervical Cancer dataset."""
    df_rfcc = df_rfcc.drop(columns=["Citology", "Schiller", "Hinselmann"])
    df_rfcc["Biopsy"] = pd.Categorical(
        df_rfcc["Biopsy"], categories=[0, 1], ordered=True
    ).rename_categories(["Healthy", "Cancer"])
    df_rfcc["IUD"] = df_rfcc["IUD"].astype(float)
    df_rfcc = df_rfcc[
        [
            "Age",
            "Number of sexual partners",
            "First sexual intercourse",
            "Num of pregnancies",
            "Smokes",
            "Smokes (years)",
            "Hormonal Contraceptives",
            "Hormonal Contraceptives (years)",
            "IUD",
            "IUD (years)",
            "STDs",
            "STDs (number)",
            "STDs: Number of diagnosis",
            "STDs: Time since first diagnosis",
            "STDs: Time since last diagnosis",
            "Biopsy",
        ]
    ]
    # Impute missing values using the most frequent value (mode)
    df_rfcc = df_rfcc.replace("?", np.nan)
    imputer = SimpleImputer(strategy="most_frequent")
    df_rfcc.iloc[:, :-1] = imputer.fit_transform(df_rfcc.iloc[:, :-1])
    df_rfcc_imputed = pd.DataFrame(
        df_rfcc, columns=df_rfcc.columns[:-1]
    )  # Drop target column from output
    df_rfcc = pd.concat([df_rfcc_imputed, df_rfcc["Biopsy"]], axis=1)
    return df_rfcc
