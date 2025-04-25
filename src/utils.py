import os
import yaml
import pandas as pd
import xgboost
import unicodedata
import numpy as np


def mode_values(values):
    try:
        return values.mode().iloc[0]
    except:
        return None


def diff_maxMin(values):
    try:
        return values.max() - values.min()
    except:
        return None


def get_datetime_column(root_path, file_name, table_name):
    all_element = None
    time_cols = None
    # Lấy cấu hình của cột datetime
    with open(os.path.join(root_path, file_name), "r") as f:
        time_cols = yaml.safe_load(f)
    f.close()

    if table_name not in time_cols.keys():
        return None, None
    else:
        time_cols = time_cols[table_name]
    
    # Lấy thông tin của cột datetime
    if type(time_cols) == dict:
        if (
            "element" in time_cols.keys()
            and "time_cols" in time_cols.keys()
        ):
            all_element = time_cols["element"]
            time_cols = time_cols["time_cols"]
    return time_cols, all_element


def get_high_corr_col(df):
    columns = list(df.columns[1:])
    high_corr_cols = list()
    for i in range(len(columns)):
        col1 = columns[i]
        if col1 not in high_corr_cols:
            for j in range(i + 1, len(columns)):
                col2 = columns[j]
                if col2 not in high_corr_cols:
                    if df[[col1, col2]].corr().iloc[0, 1] > 0.8:
                        high_corr_cols.append(col2)
    return high_corr_cols


def get_low_up_iqr(values, threshold):
    quantile25 = values.quantile(0.25)
    quantile75 = values.quantile(0.75)
    
    iqr = quantile75 - quantile25
    lower = quantile25 - threshold * iqr
    upper = quantile75 + threshold * iqr
    return lower, upper


def count_outlier(df, columns=None, threshold=1.5):
    df_outlier = pd.DataFrame()
    if columns is None:
        columns = df.columns
    for col in columns:
        lower, upper = get_low_up_iqr(df[col], threshold)
        df_outlier = pd.concat(
            [
                df_outlier,
                pd.DataFrame(
                    {
                        "column": [col],
                        "count": [((df[col] < lower) | (df[col] > upper)).sum()],
                        "percent": [
                            100 * ((df[col] < lower) | (df[col] > upper)).sum()
                            / _df_train.shape[0]
                        ],
                    }
                )
            ],
            ignore_index=True
        )
    return df_outlier


def get_fold(df, fold, lable_name, map_label=None):
    df_train = df[~df["fold"].eq(fold)].drop(columns="fold").copy()
    df_valid = df[df["fold"].eq(fold)].drop(columns="fold").copy()

    X_train = df_train.drop(columns=lable_name).reset_index(drop=True)
    X_valid = df_valid.drop(columns=lable_name).reset_index(drop=True)

    y_train = df_train[lable_name].reset_index(drop=True)
    y_valid = df_valid[lable_name].reset_index(drop=True)
    if map_label is None:
        map_label = {
            "good": 0,
            "bad": 1,
        }

    y_train = y_train.map(map_label)
    y_valid = y_valid.map(map_label)
    return X_train, X_valid, y_train, y_valid


def get_train_test(df_train, df_test, lable_name, map_label=None):
    X_train = df_train.drop(columns=[lable_name, "fold"]).reset_index(drop=True)
    X_test = df_test.drop(columns=lable_name).reset_index(drop=True)

    y_train = df_train[lable_name].reset_index(drop=True)
    y_test = df_test[lable_name].reset_index(drop=True)

    if map_label is None:
        map_label = {
            "good": 0,
            "bad": 1,
        }

    y_train = y_train.map(map_label)
    y_test = y_test.map(map_label)

    return X_train, X_test, y_train, y_test


def select_feature_importance(X_train, y_train, threshold_importance, **model_params):
    xgb = xgboost.XGBClassifier(**model_params)
    xgb.fit(X_train, y_train)
    
    df_importance = pd.DataFrame(
        {
            "column": X_train.columns,
            "importance": xgb.feature_importances_,
        }
    )
    
    importance_cols = df_importance.loc[df_importance["importance"] > threshold_importance, "column"].values

    return importance_cols


def statistic_valid_score(df_result):
    aggs = ["mean", "std", "max", "min"]
    
    df_sumary = pd.DataFrame(index=["Train", "Validated"])
    for agg in aggs:
        df_sumary = pd.merge(
            left=df_sumary,
            right=pd.DataFrame(
                    {
                        "Train": df_result[["KS", "AUC"]].T.filter(like="Train").T.agg(agg),
                        "Validated": df_result[["KS", "AUC"]].T.filter(like="Test").T.agg(agg),
                    }
            ).T.add_prefix(f"{agg}_"),
            left_index=True,
            right_index=True,
        )
    df_sumary["Size"] = df_result["Size"].iloc[:2].values
    return df_sumary


#############
# FORMAT_DATA
#############
def format_bin(bin_value):
    """
    Formats a bin value string into a more human-readable form.
    
    Parameters:
        bin_value (str): A string representing a bin value.
        
    Returns:
        str: A formatted string representing the bin value in a human-readable form.
    """
    bin_value = bin_value.strip()  # Remove any unwanted spaces

    if bin_value.lower() == "missing":
        return "Missing"

    elif bin_value.startswith("[-inf"):
        # Handle the special case with -inf
        try:
            upper = bin_value.split(',')[1].strip().strip(')')
            upper_float = float(upper)
            upper_rounded = f"{upper_float / 1e6:.2f} Mil" if upper_float >= 1e6 else f"{upper_float:,.2f}"
            return f"[-∞, {upper_rounded})"
        except ValueError:
            return bin_value  # Fallback to original if conversion fails

    elif bin_value.endswith("inf)"):
        # Handle the case with "inf"
        try:
            lower = bin_value.split(',')[0].strip().strip('[')
            lower_float = float(lower)
            lower_rounded = f"{lower_float / 1e6:.2f} Mil" if lower_float >= 1e6 else f"{lower_float:,.2f}"
            return f"[{lower_rounded}, ∞)"
        except ValueError:
            return bin_value  # Fallback to original if conversion fails

    else:
        # Handle finite numeric intervals
        try:
            lower, upper = bin_value.strip("[]").split(',')
            lower, upper = lower.strip(), upper.strip().strip(')')
            lower_float, upper_float = float(lower), float(upper)

            lower_str = f"{lower_float / 1e6:.2f} Mil" if lower_float >= 1e6 else f"{lower_float:,.2f}"
            upper_str = f"{upper_float / 1e6:.2f} Mil" if upper_float >= 1e6 else f"{upper_float:,.2f}"

            return f"[{lower_str}, {upper_str})"
        except ValueError:
            return bin_value  # Fallback for unexpected formats
            

def unicode_text(text):
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    ).lower().replace(" ", "")


def map_province(province, df_region):
    for value in df_region["province_name"].values:
        if value in province:
            return df_region[df_region["province_name"].eq(value)]["region_name"].iloc[0]
    return "Không xác định"

    
###########
# CACULATE
###########
def caculate_crosstab(df: pd.DataFrame, columns: list, sort_values = None, is_transpose=True):
    df_cross_count = pd.crosstab(df[columns[0]], df[columns[1]])
    if sort_values is not None:
        df_cross_count = df_cross_count[sort_values[1]].T
        df_cross_count = df_cross_count[sort_values[0]].T
        
    if df_cross_count.index.shape[0] < df_cross_count.columns.shape[0] and is_transpose:
        df_cross_count = df_cross_count.T
        columns = [columns[1], columns[0]]
    
    df_cross_count.index.name = None
    df_cross_count.columns.name = None

    df_cross_percent_all = 100 * df_cross_count.div(df_cross_count.sum().sum())

    df_cross_percent_horizontal = 100 * df_cross_count.div(df_cross_count.sum(axis=1), axis=0)

    df_cross_percent_vertical = 100 * df_cross_count.div(df_cross_count.sum(axis=0), axis=1)
    
    return df_cross_count, df_cross_percent_all, df_cross_percent_horizontal, df_cross_percent_vertical, columns


############
# GET CONFIG
############
def get_config_bin(root_path, df, ignore_cols=None):
    with open(os.path.join(root_path, "config/bin_cols.yaml"), "r") as f:
        bin_cols = yaml.safe_load(f)
    f.close()
    
    breaks_list = dict()
    for col in df.columns:
        if col in ignore_cols:
            continue
        for config_col in bin_cols.keys():
            if config_col in col:
                breaks_list[col] = bin_cols[config_col]
        if (df[col].dtype == "object"):
            breaks_list[col] = list(df[col].unique())
        if (df[col].unique().shape[0] > 2) and (df[col].unique().shape[0] <= 10):
            breaks_list[col] = list(df[col].sort_values().unique())[1:-1]
    return breaks_list

##############
# EXPLORE DATA
##############
def overview_table(root_path: str=None, file_name: str=None, df: pd.DataFrame=None):
    """
    Tạo bảng tổng quan về dữ liệu, bao gồm:
    - Số lượng giá trị trùng lặp (`Count_duplicate`)
    - Số lượng giá trị bị thiếu (`Count_missing`)
    - Tỷ lệ phần trăm giá trị bị thiếu (`Percent_missing`)
    - Số lượng giá trị duy nhất (`Count_distinct`)
    - Số lượng giá trị bằng 0 (`Count_zero`, chỉ áp dụng cho kiểu số)
    - Tỷ lệ phần trăm giá trị bằng 0 (`Percent_zero`, chỉ áp dụng cho kiểu số)
    
    Args:
        root_path (str, optional): Đường dẫn gốc của file dữ liệu (nếu cần load từ file CSV).
        file_name (str, optional): Tên file CSV để đọc dữ liệu.
        df (pd.DataFrame, optional): DataFrame đầu vào (nếu đã có sẵn).
    
    Returns:
        pd.DataFrame: DataFrame chứa thông tin tổng quan về các cột.
    """
    
    # Nếu có đường dẫn và tên file, đọc dữ liệu từ CSV
    if (root_path is not None) and (file_name is not None):
        df = pd.read_csv(os.path.join(root_path, f"data/{file_name}"))
    
    df_overview = []  # Dùng list để tối ưu hiệu suất khi tạo DataFrame
    
    for col in df.columns:
        col_type = str(df[col].dtype)
        
        col_info = {
            "Column": col,
            "Count_duplicate": df[col].duplicated().sum(),
            "Count_missing": df[col].isna().sum(),
            "Percent_missing": 100 * df[col].isna().sum() / df.shape[0],
            "Count_distinct": df[col].nunique(),
        }
        
        if "int" in col_type or "float" in col_type:
            col_info.update({
                "Count_zero": df[col].eq(0).sum(),
                "Percent_zero": 100 * df[col].eq(0).sum() / df.shape[0]
            })
        else:
            col_info.update({"Count_zero": None, "Percent_zero": None})
        
        df_overview.append(col_info)
    
    return pd.DataFrame(df_overview)


def create_bins(value, num_bin = 5, outline=0.05):
    bins = list(
        np.linspace(
            value.quantile(outline), value.quantile(1 - outline), num_bin - 1
        )
    )
    result = list()
    for val in list(dict.fromkeys([value.min()] + bins + [value.max()])):
        if int(val) not in result:
            result.append(int(val))
    return result
    

def statistic_bins_with_time(df, value_column, time_column, num_bin=5):
    df_result = pd.DataFrame()
    bins = create_bins(df[value_column], num_bin=num_bin)
    times = df[time_column].sort_values().unique()
    for time in times:
        _df_temp = pd.DataFrame(
            pd.cut(df[df[time_column].eq(time)][value_column], bins=bins)
            .value_counts().sort_index()
        ).rename(columns={"count": time})
        if df_result.empty:
            df_result = _df_temp
        else:
            df_result = pd.merge(
                left=df_result, right=_df_temp, left_index=True, right_index=True
            )
    return df_result[times]
    

def statistic_category_with_time(df, value_column, time_column):
    df_result = pd.DataFrame()
    times = df[time_column].sort_values().unique()
    for time in times:
        _df_temp = pd.DataFrame(
            df[df[time_column].eq(time)][value_column].value_counts()
        ).rename(columns={"count": time})
        if df_result.empty:
            df_result = _df_temp
        else:
            df_result = pd.merge(
                left=df_result, right=_df_temp, left_index=True, right_index=True
            )
    return df_result[times]


def caculate_psi(df, value_column=None):
    df_obs = df / df.sum(axis=0)
    df_ref = df_obs.shift(axis=1)
    if value_column is None:
        value_column = df.index.name
    return pd.DataFrame(
        (
            (df_ref - df_obs) * (df_ref / df_obs).apply(np.log)
        ).sum(axis=0)
    ).rename(columns={0: value_column}).T
    

def caculate_psi(df, value_column=None):
    df_obs = df / df.sum(axis=0)
    df_ref = df_obs.shift(axis=1)
    if value_column is None:
        value_column = df.index.name
    return pd.DataFrame(
        (
            (df_ref - df_obs) * (df_ref / df_obs).apply(np.log)
        ).sum(axis=0)
    ).rename(columns={0: value_column}).T


def creae_df_psi(df, freq, time_column):
    df_psi = pd.DataFrame()
    _df = df.copy()
    _df[time_column] = _df[time_column].dt.to_period(freq)
    _df = _df.sort_values(time_column)
    cols = [col for col in _df.columns if col != time_column]
    for col in cols:
        try:
            if df[col].dtype == "object":
                df_statistic_with_time = statistic_category_with_time(
                    df=_df, value_column=col, time_column=time_column,
                )
            else:
                df_statistic_with_time = statistic_bins_with_time(
                    df=_df, value_column=col, time_column=time_column
                )
                
            df_psi = pd.concat(
                [df_psi, caculate_psi(df=df_statistic_with_time)]
            )
        except:
            print(f"Fail for {col}")
    return df_psi


def stastistic_category(df, column):
    df_val_count = pd.DataFrame(df[column].value_counts())
    df_val_count["percent"] = 100 * df_val_count["count"] / df_val_count["count"].sum()
    return df_val_count