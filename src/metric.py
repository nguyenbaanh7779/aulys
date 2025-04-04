import pandas as pd
import numpy as np


def caculate_lift(y_true, y_probPred, type_class=1, n_bins=10):
    if type_class:
        df_lift = pd.DataFrame(
            {
                "y_true": y_true,
                "y_scores": y_probPred,
            }
        ).sort_values("y_scores", ascending=False).reset_index(drop=True)
    else:
        df_lift = pd.DataFrame(
            {
                "y_true": y_true,
                "y_scores": 1 - y_probPred,
            }
        ).sort_values("y_scores", ascending=False).reset_index(drop=True)

        df_lift["y_true"] = df_lift["y_true"].map({
                0: 1, 1: 0
        })
    df_lift["quantile"] = pd.qcut(df_lift.index, n_bins, labels=False)
    
    df_lift = df_lift.groupby("quantile").agg(
        positive=("y_true", "sum")
    ).reset_index()
    
    df_lift["cumulative_positive"] = df_lift["positive"].cumsum()
    df_lift["cumulative_rate"] = df_lift["cumulative_positive"] / df_lift["positive"].sum()
    df_lift["random_rate"] = (df_lift.index + 1) / len(df_lift)
    df_lift["lift"] = df_lift["cumulative_rate"] / df_lift["random_rate"]
    
    return df_lift


def caculate_gain(y_true, y_probPred):
    df_gain = pd.DataFrame(
        {
            "y_true": y_true,
            "y_score": y_probPred,
        }
    ).sort_values("y_score", ascending=False).reset_index(drop=True)
    
    df_gain["cumulative"] = df_gain["y_true"].cumsum()
    
    df_gain["percent_sample"] = np.arange(1, len(df_gain) + 1) / len(df_gain)
    df_gain["gain"] = df_gain["cumulative"] / df_gain["y_true"].sum()
    return df_gain