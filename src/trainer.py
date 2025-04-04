import pandas as pd
import scorecardpy as sc
from tqdm import tqdm


def create_woe_bin(df, label_name, feature_columns, batch_size=25, map_lable=None, breaks_list=None):
    num_batch = int(len(feature_columns) / batch_size)
    df_bin = pd.DataFrame()

    df_feature = df[[label_name] + feature_columns].copy()
    df_feature = df_feature.drop_duplicates()
    if map_lable is not None:
        df_feature[label_name] = df_feature[label_name].map(map_lable)
    # return df_feature
    for i in range(0, num_batch):
        print(f"Run with range ({i * batch_size} - {(i + 1) * batch_size})")
        df_select = df_feature[[label_name] + feature_columns[i * batch_size: (i+1) * batch_size]].copy()
        # df_select = sc.var_filter(dt=df_select, y=lable_name)
        bins = sc.woebin(dt=df_select, y=label_name, breaks_list=breaks_list)
        for col in bins.keys():
            df_bin = pd.concat([df_bin, bins[col]])
    
    print(f"Run with range ({num_batch*batch_size} - {len(feature_columns)})")
    df_select = df_feature[[label_name] + feature_columns[num_batch*batch_size:]].copy()
    # df_select = sc.var_filter(dt=df_select, y=lable_name)
    bins = sc.woebin(dt=df_select, y=label_name, breaks_list=breaks_list)
    for col in bins.keys():
        df_bin = pd.concat([df_bin, bins[col]])
    return df_bin