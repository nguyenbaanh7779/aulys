import os
import json
import ast
import yaml
import pandas as pd
import pyspark.sql.functions as F
import pyspark.sql.types as T
import scorecardpy as sc
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# from sliced_master.sliced.sir import SlicedInverseRegression
import config
import src.trainer as trainer
import src.utils as utils


# if value is not None:
def tranform_datetimeFormat(value):
    try:
        value = str(int(value))
        if len(value) == 7:
            value = f"0{value}"
        return value
    except:
        return None


def restruct_dict_form(values):
    try:
        return values[1:-1].replace('""', '"')
    except:
        return values


def convert_list(values):
    try:
        values = ast.literal_eval(values.replace("null", '"None"'))
        return [json.dumps(value) for value in values]
    except:
        return values

#################################
# GET DATA FROM DICTIONARY COLUMN
#################################
def get_column_from_dict(data_dict, table_name=None):
    """
    Get columns from dict data type column
    Parameter:
        - data_dict: a value of dict data type column
        - table_name: the road has reached dict contain columns
    Return:
        - list of column
    """
    data_dict_temp = data_dict
    if table_name is not None:
        for key in table_name.split("."):
            data_dict_temp_v2 = data_dict_temp[key]
            data_dict_temp = data_dict_temp_v2
    return data_dict_temp.keys()


def get_data_from_dict(
    sdf, root_path, column_name, table_name=None, drop_columns=None, is_check=False, file_name=None
):
    # get data from dict
    data_dict = json.loads(sdf.select(column_name).first()[0])
    for col_name in get_column_from_dict(data_dict=data_dict, table_name=table_name):
        if table_name is None:   
            sdf = sdf.withColumn(
                col_name, 
                F.get_json_object(F.col(column_name), f"$.{col_name}")
            )
        else:
            sdf = sdf.withColumn(
                col_name, 
                F.get_json_object(F.col(column_name), f"$.{table_name}.{col_name}")
            )
            
    # drop unnecessary column
    sdf = sdf.drop(column_name)
    if drop_columns is not None:
        sdf = sdf.drop(*drop_columns)

    # type of get_data
    if is_check:
        df = sdf.limit(100).toPandas()
        return df
    else:
        # set name of file to save
        if file_name is None:
            if table_name is None:
                file_name = column_name
            else:
                file_name = table_name.replace(".", "_")
        print(f"Save file into {os.path.join(root_path, f"data/raw/{config.DATE_KEY}/{file_name}.csv")}")
        # return sdf
        df = sdf.toPandas()
        if not os.path.exists(os.path.join(root_path, f"data/raw/{config.DATE_KEY}")):
            os.makedirs(os.path.join(root_path, f"data/raw/{config.DATE_KEY}"))
        df.to_csv(os.path.join(root_path, f"data/raw/{config.DATE_KEY}/{file_name}.csv"), index=False)
        # return None


def explode_data_from_dict(sdf, table_name, column_name):
    """
    Explode list format data
    Example:
    Original data:
    +-+------+
    |1|[a, b]|
    +-+------+
    |2|[c, d]|
    +-+------+
    After pass this function
    +-+-+
    |1|a|
    +-+-+
    |1|b|
    +-+-+
    |2|c|
    +-+-+
    |2|d|
    +-+-+

    Parameter:
        - sdf: spark dataframe
        - table_name: the path to the list data
        - column_name: the name of column contains dictionary data
    Return:
        - spark dataframe after explode
    """
    # Select data
    _column_name = table_name.replace(".", "_")
    sdf_table = sdf.select("*")
    sdf_table = sdf_table.withColumn(_column_name, F.get_json_object(F.col(column_name), f"$.{table_name}"))
    drop_cols = list()
    sdf_table = sdf_table.drop(column_name)
    
    # Convert string to list in data
    udf_func = F.udf(convert_list, T.ArrayType(T.StringType()))
    sdf_table = sdf_table.withColumn(_column_name, udf_func(F.col(_column_name)))
    
    # Exlode data
    sdf_table = sdf_table.withColumn(_column_name, F.explode(F.col(_column_name)))
    return sdf_table
    

def get_data_from_dict_v2(sdf, table_name, column_name, root_path, file_name=None, is_check = False):
    """
    Save data have 1_n structure

    Parameter:
        - sdf: pyspark dataframe
        - table_name: the road has reached the data in the JSON file
        - column_name: the name of the data column is JSON
        - root_path: path of root folder
        - file_name: the name of data file to save
        - is_check: if want to check output of this function

    Return:
        dataframe: if is_check = True
        None: if is_check = False
    """
    sdf_table = explode_data_from_dict(sdf, table_name, column_name)
    _column_name = table_name.replace(".", "_")
    
    # take columns with JSON data type
    dict_cols = list()
    for col in json.loads(sdf_table.select(_column_name).first()[0]).keys():
        if col in map_table.keys():
            dict_cols.append(col)

    # call log when file name is none
    if file_name is None:
        print(f"getting_data of table {table_name}")

    # save data in the original path but do not contain JSON data type columns
    display(
        get_data_from_dict(
            sdf=sdf_table,
            root_path=root_path,
            column_name=_column_name,
            # table_name="InquiredOperation",
            drop_columns=dict_cols,
            is_check=is_check,
            file_name=file_name
        )
    )

    # save data of JSON data type columns if these columns exist
    if dict_cols != []:
        for col in dict_cols:
            if "1_1" in map_table[col].keys():
                for _table_name in map_table[col]["1_1"]:
                    print(f"Getting data of {table_name}.{_table_name} ...")
                    display(
                        get_data_from_dict(
                            sdf=sdf_table,
                            root_path=root_path,
                            column_name=_column_name,
                            table_name=_table_name,
                            is_check=is_check,
                            file_name = f"{table_name}.{_table_name}".replace(".", "_")
                        )   
                    )
            if "1_n" in map_table[col].keys():
                for _table_name in map_table[col]["1_n"]:
                    print(f"Getting data of {table_name}.{_table_name} ...")
                    get_data_from_dict_v2(
                        sdf=sdf_table,
                        root_path = root_path,
                        table_name=_table_name,
                        column_name=_column_name,
                        file_name=f"{table_name}.{_table_name}".replace(".", "_"),
                        is_check=is_check
                    )


########################
# GET DATA FROM HUE PATH
########################
def get_sample_data(spark, root_path, table_name):
    # Load path hue of table
    with open(os.path.join(root_path, "config/path_hue.yaml"), "r") as f:
        hue_path = yaml.safe_load(f)[table_name]
    f.close()
    
    # Read data from hue path
    if hue_path[-1] == "=":
        hue_path = f"{hue_path}{config.TODAY}"
    dfs = (
         spark.read.parquet(hue_path)
    )
    
    # Load time cols of table
    with open(os.path.join(root_path, "config/time_cols.yaml"), "r") as f:
        time_cols = yaml.safe_load(f)
        time_cols = time_cols.get(table_name, [])
    f.close()
    
    # convert time cols dtype from datetime to string
    for col in time_cols:
        dfs = dfs.withColumn(col, F.col(col).cast("string"))
        
    # Save file
    df = dfs.toPandas()
    print(f"Save file into {os.path.join(root_path, f"data/raw/{config.TODAY}/{table_name}.csv")}")
    df.to_csv(os.path.join(root_path, f"data/raw/{config.TODAY}/{table_name}.csv"), index=False)
    return df


#######################
# GET PROCESSED DATASET
#######################
def get_dataset(root_path, group_tables=None, use_cols=None, drop_cols=None, **kwargs):
    with open(os.path.join(root_path, "config/map_table.yaml"), "r") as f:
        map_table = yaml.safe_load(f)
    table_names = list()
    for group_table in group_tables:
        table_names += map_table[group_table]
    
    df = pd.read_csv(os.path.join(
            root_path, f"data/processed/{config.TODAY}/{config.UTM_SOURCE}/sample.csv"
    )).set_index("contract_id")
    
    for table_name in table_names:
        file_name = table_name.replace(".", "_")
        if not os.path.exists(os.path.join(
                root_path, f"data/processed/{config.TODAY}/{config.UTM_SOURCE}/{file_name}.csv"
            )):
            continue
        print(f"Reading table {file_name} ...")
        df = pd.merge(
            left=df,
            right=pd.read_csv(os.path.join(
                root_path, f"data/processed/{config.TODAY}/{config.UTM_SOURCE}/{file_name}.csv"
            )).set_index("contract_id").add_prefix(f"{file_name}_"),
            left_index=True,
            right_index=True,
            how="left"
        )

    if use_cols is not None:
        df = df[["label"] + used_cols]
    if drop_cols is not None:
        df = df.drop(columns=drop_cols)
    return df
    

###################
# AGGREGATE FEATURE
###################
def agg_num_fea(df, idx_col, num_cols):
    """
    Tổng hợp các đặc trưng dạng số bằng các phép thống kê: 
    - Trung bình (`mean`)
    - Giá trị lớn nhất (`max`)
    - Giá trị nhỏ nhất (`min`)
    - Tổng (`sum`)
    - Trung vị (`median`)
    - Hiệu số giữa max và min (`diff`)
    
    Args:
        df (pd.DataFrame): Dữ liệu đầu vào
        idx_col (str): Cột index (ID nhóm)
        num_cols (list): Danh sách các cột số để tổng hợp

    Returns:
        pd.DataFrame: DataFrame chứa các feature tổng hợp
    """
    # Tính toán các giá trị thống kê
    df_agg = df.groupby(idx_col)[num_cols].agg(["mean", "max", "min", "sum", "median"])
    df_agg.columns = [f"{stat}_{col}" for col, stat in df_agg.columns]
    df_agg = df_agg.reset_index()

    # Tính giá trị chênh lệch giữa max và min
    df_diff = df.groupby(idx_col)[num_cols].agg(lambda x: x.max() - x.min())
    df_diff = df_diff.add_prefix("diff_").reset_index()

    return df_agg.merge(df_diff, on=idx_col)

def agg_cat_fea(df, idx_col, cat_cols):
    """
    Tổng hợp các đặc trưng dạng phân loại bằng các phép thống kê:
    - Số lượng giá trị khác nhau (`nunique`)
    - Giá trị xuất hiện nhiều nhất (`mode`)
    - Tần suất xuất hiện của từng giá trị trong cột phân loại
    
    Args:
        df (pd.DataFrame): Dữ liệu đầu vào
        idx_col (str): Cột index (ID nhóm)
        cat_cols (list): Danh sách các cột phân loại để tổng hợp

    Returns:
        pd.DataFrame: DataFrame chứa các feature tổng hợp dạng phân loại
    """
    df_agg = pd.DataFrame(index=df[idx_col].unique())
    df_agg.index.name = idx_col
    
    for col in cat_cols:
        if df[col].nunique() < 2:
            continue

        # Đếm số lượng giá trị duy nhất
        df_agg[f"nunique_{col}"] = df.groupby(idx_col)[col].nunique()
        
        # Lấy mode (giá trị xuất hiện nhiều nhất)
        df_agg[f"mode_{col}"] = df.groupby(idx_col)[col].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
        
        # Tính tần suất xuất hiện của từng giá trị trong cột
        freq_df = df.groupby([idx_col, col]).size().unstack(fill_value=0)
        freq_df.columns = [f"{col}_{val}_times" for val in freq_df.columns]
        df_agg = df_agg.merge(freq_df, left_index=True, right_index=True, how="left")
    
    return df_agg.reset_index()

def agg_fea(df, idx_col, num_cols, cat_cols):
    """
    Tích hợp các feature số và phân loại thành một DataFrame duy nhất
    
    Args:
        df (pd.DataFrame): Dữ liệu đầu vào
        idx_col (str): Cột index (ID nhóm)
        num_cols (list): Danh sách các cột số
        cat_cols (list): Danh sách các cột phân loại

    Returns:
        pd.DataFrame: DataFrame chứa tất cả các feature tổng hợp
    """
    agg_num = agg_num_fea(df, idx_col, num_cols)
    agg_cat = agg_cat_fea(df, idx_col, cat_cols)
    
    return pd.concat([agg_num.set_index(idx_col), agg_cat.set_index(idx_col)], axis=1).reset_index()


################################
# CONVERT TO THE RIGHT DATA TYPE
################################
def convert_datetime(df, time_cols, all_element):
    # Khởi tạo cột date time
    if all_element is not None:
        for i, time_col in enumerate(time_cols):
            elements = all_element[i]
            df[time_col] = df[elements[0]].astype(str)
            for element in elements[1:]:
                df[time_col] = df[time_col] + "-" + df[element].astype(str)
            df[time_col] = pd.to_datetime(df[time_col], format="%Y-%m")
            df = df.drop(columns=elements)
    else:
        if type(time_cols) == list:
            for col in time_cols:
                df[col] = df[col].apply(tranform_datetimeFormat)
                df[col] = pd.to_datetime(df[col], format="%d%m%Y")
    return df 

################
# PROCESSED DATA
################
def create_woe_feature(df, fold):
    # split train, valid
    _df_train = df[~df["fold"].eq(fold)].drop(columns="fold").copy()
    _df_valid = df[df["fold"].eq(fold)].drop(columns="fold").copy()

    _df_train = _df_train.reset_index(drop=True)
    _df_valid = _df_valid.reset_index(drop=True)
    
    columns = list(_df_train.columns[1:])

    # make woe bin
    df_bin = trainer.create_woe_feature(df=_df_train, columns=columns)
    _df_train = sc.woebin_ply(dt=_df_train, bins=df_bin, print_info=False)
    _df_valid = sc.woebin_ply(dt=_df_valid, bins=df_bin, print_info=False)

    # drop high corr columns
    drop_cols = utils.get_high_corr_col(df=_df_train)
    _df_train = _df_train.drop(columns=drop_cols)
    _df_valid = _df_valid.drop(columns=drop_cols)

    # set X, y for model
    X_train = _df_train.drop(columns = "dpd_30")
    y_train = _df_train["dpd_30"]
    X_valid = _df_valid.drop(columns = "dpd_30")
    X_valid = X_valid[X_valid.columns].fillna(0)
    y_valid = _df_valid["dpd_30"]

    return X_train, y_train, X_valid, y_valid


def process_categorical(X_train, X_test, **kwargs):
    cat_cols = list(
        X_train.dtypes[
            ((X_train.dtypes.eq("object")) | (X_train.dtypes.eq("bool")))
        ].index
    )
    if len(cat_cols) == 0:
        return None, None
        
    X_train = X_train[cat_cols].copy()
    X_test = X_test[cat_cols].copy()
    
    # fill nan cho các cột categorical
    X_train[cat_cols] = X_train[cat_cols].fillna("Unknow")
    X_test[cat_cols] = X_test[cat_cols].fillna("Unknow")

    use_encode = kwargs.get("use_encode", True)
    if use_encode:
        # bool encode
        bool_cols = list()
        for col in X_train.columns:
            if X_train[col].isna().sum() > 0:
                X_train[col] = X_train[col].astype(str)
                X_test[col] = X_test[col].astype(str)
            if X_train[col].dtype == "bool":
                X_train[col] = X_train[col].astype(int)
                X_test[col] = X_test[col].astype(int)
                bool_cols.append(col)
    
        # one hot encode
        one_hot_cols = list()
        for col in X_train.columns:
            if col not in bool_cols:
                one_hot_cols.append(col)
        X_train = pd.merge(
            left=X_train[bool_cols],
            right=pd.get_dummies(X_train[one_hot_cols], columns=one_hot_cols).astype(int),
            left_index=True,
            right_index=True
        )
        X_test = pd.merge(
            left=X_test[bool_cols],
            right=pd.get_dummies(X_test[one_hot_cols], columns=one_hot_cols),
            left_index=True,
            right_index=True
        )
        for col in X_train.columns:
            if col not in X_test.columns:
                X_test[col] = 0
    else:
        X_train = X_train.astype("category")
        X_test = X_test.astype("category")
    X_test = X_test[X_train.columns]
    return X_train, X_test


def process_numerical(X_train, X_test, **kwargs):
    #choice numerical columns
    num_cols = list(
        X_train.dtypes[
            ~((X_train.dtypes.eq("object")) | (X_train.dtypes.eq("bool")))
        ].index
    )

    if len(num_cols) == 0:
        return None, None
    
    X_train = X_train[num_cols].copy()
    X_test = X_test[num_cols].copy()
    remove_cols = list()

    limit_missing = kwargs.get("limit_missing", None)
    if limit_missing is not None:
        df_missing = X_train[num_cols].isna().sum() / X_train.shape[0]
        remove_cols = remove_cols + list(df_missing[df_missing > limit_missing].index)
    
    limit_outlier = kwargs.get("limit_outlier", None)
    threshold_iqr = kwargs.get("threshold_iqr", 1.5)
    if limit_outlier is not None:
        df_outlier = utils.count_outlier(df=X_train, columns=num_cols, threshold=threshold_iqr)
        remove_cols = remove_cols + list(df_outlier[df_outlier["percent"] > limit_outlier]["column"].values)

    X_train = X_train.drop(columns=remove_cols)
    X_test = X_test.drop(columns=remove_cols)
    # fill outlier
    for col in X_train.columns:
        lower, upper = utils.get_low_up_iqr(X_train[col], threshold_iqr)
        X_train.loc[X_train[col] < lower, col] = lower
        X_train.loc[X_train[col] > upper, col] = upper
        X_test.loc[X_test[col] < lower, col] = lower
        X_test.loc[X_test[col] > upper, col] = upper
    
    # fill missing
    for col in X_train.columns:
        type_fill_missing = kwargs.get("type_fill_missing", "mean")
        if type_fill_missing == "min":
            fill_value = X_train[col].min() - 1
        elif type_fill_missing == "max":
            fill_value = X_train[col].max() + 1
        else:
            fill_value = X_train[col].mean()
        
        X_train[col] = X_train[col].fillna(fill_value)
        X_test[col] = X_test[col].fillna(fill_value)
    
    # scaler feature
    use_scaler = kwargs.get("use_scaler", True)
    if use_scaler:
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train[X_train.columns] = scaler.transform(X_train)
        X_test[X_test.columns] = scaler.transform(X_test)
        X_test = X_test[X_train.columns]
    return X_train, X_test


def create_pca(X_train, X_test, **pca_params):
    X_train_pca = X_train.copy()
    X_test_pca = X_test.copy()
    decomposition = PCA(**pca_params)
    decomposition.fit(X_train_pca)
    X_train_pca = decomposition.transform(X_train_pca)
    X_test_pca = decomposition.transform(X_test_pca)
    pc_columns = [f"PC_{i + 1}" for i in range(X_train_pca.shape[1])]
    X_train_pca = pd.DataFrame(data=X_train_pca, columns=pc_columns)
    X_test_pca = pd.DataFrame(data=X_test_pca, columns=pc_columns)
    return X_train_pca, X_test_pca

def create_sir(X_train, X_test, y_train=None, **sir_params):
    sir = SlicedInverseRegression(n_directions=5)
    check = sir.fit(X_train, y_train)
    X_train_sir = sir.transform(X_train)
    X_test_sir = sir.transform(X_test)
    pc_columns = [f"PC_{i + 1}" for i in range(X_train_sir.shape[1])]
    X_train_sir = pd.DataFrame(data=X_train_sir, columns=pc_columns)
    X_test_sir = pd.DataFrame(data=X_test_sir, columns=pc_columns)
    return X_train_sir, X_test_sir


def create_reduce(X_train, X_test, y_train=None, **reduce_params):
    reduce_method = reduce_params.get("method", "PCA")
    del reduce_params["method"]
    
    X_train_non_importance = None
    X_test_non_importance = None
    if reduce_method == "PCA":
        X_train_non_importance, X_test_non_importance = create_pca(
            X_train=X_train,
            X_test=X_test,
            **reduce_params
        )
    elif reduce_method == "SIR":
        X_train_non_importance, X_test_non_importance = create_sir(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            **reduce_params
        )
        
    return X_train_non_importance, X_test_non_importance


def process_non_importance(X_train, X_valid, importance_cols, y_train=None, **reduce_params):
    no_importance_cols = X_train.drop(columns=importance_cols).columns
    X_train_importance = X_train[importance_cols]
    X_valid_importance = X_valid[importance_cols]
    reduce_method = reduce_params.get("method", "PCA")
    
    X_train_non_importance, X_valid_non_importance = create_reduce(
        X_train=X_train[no_importance_cols],
        X_test=X_valid[no_importance_cols],
        y_train=y_train,
        **reduce_params,
    )
    
    X_train = pd.merge(
        left=X_train_importance,
        right=X_train_non_importance,
        left_index=True,
        right_index=True,
    )
    
    X_valid = pd.merge(
        left=X_valid_importance,
        right=X_valid_non_importance,
        left_index=True,
        right_index=True,
    )
    return X_train, X_valid


def process_data(X_train, X_test, y_train=None, **processed_params):
    model_params = processed_params["model_params"]
    pca_params = processed_params["pca_params"]
    
    use_cat_col = processed_params.get("use_cat_col", True)
    use_num_col = processed_params.get("use_num_col", True)
    
    X_train_num, X_test_num = process_numerical(X_train, X_test, **processed_params)
    if (X_train_num is None) and (X_test_num is None):
        use_num_col = False
        
    X_train_cat, X_test_cat = process_categorical(X_train, X_test, **processed_params)
    if (X_train_cat is None) and (X_test_cat is None):
        use_cat_col = False

    use_reduce = processed_params.get("use_reduce", False)
    if use_reduce:
        X_train_num, X_test_num = create_reduce(
            X_train=X_train_num,
            X_test=X_test_num,
            y_train=y_train,
            **reduce_params,
        )

    if use_cat_col and use_num_col:
        X_train = pd.merge(
            left=X_train_cat, right=X_train_num,
            left_index=True, right_index=True,
        )
        X_test = pd.merge(
            left=X_test_cat, right=X_test_num,
            left_index=True, right_index=True,
        )
        X_test = X_test[X_train.columns]
    elif use_cat_col:
        X_train = X_train_cat
        X_test = X_test_cat
        X_test = X_test[X_train.columns]
    elif use_num_col:
        X_train = X_train_num
        X_test = X_test_num
        X_test = X_test[X_train.columns]
    else:
        raise "No columns are used"

    if processed_params.get("use_importance", True):
        threshold_importance = processed_params.get("threshold_importance", 0)
        importance_cols = utils.select_feature_importance(X_train, y_train, threshold_importance, **model_params)
        if processed_params.get("process_non_importance", True):
            X_train, X_test = process_non_importance(X_train, X_test, importance_cols, y_train=y_train, **pca_params)
        else:
            X_train = X_train[importance_cols]
            X_test = X_test[importance_cols]
    
    return X_train, X_test