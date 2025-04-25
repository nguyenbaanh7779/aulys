import lets_plot as lp
import pandas as pd
import numpy as np
import sklearn.metrics as M
from IPython.display import display, Markdown

import src.metric as metric
import src.utils as utils
import src.data as data


def letsplot_bar(group: str | None = None):
    return lp.geom_bar(
        lp.aes(fill=group, group=group),
        stat="identity", position="dodge"
    )


def letsplot_line(
    group: str | None = None, color: str | None = None, display_point: bool = False
):
    result = lp.geom_line(lp.aes(group=group, color=group, linetype=group), color=color)
    if display_point:
        result += lp.geom_point(size=4, color=color)
    return result


def letsplot_heatmap(value: str, label_format: str | None = None):
    return (
        lp.geom_tile(lp.aes(fill=value), size=0.5)
        + lp.geom_text(lp.aes(label=value), color="black", label_format=label_format)
        + lp.scale_fill_gradient(high="#FF6600", low="#FFFFCC")
    )


def letsplot_density(group):
    return lp.geom_density(lp.aes(group=group, color=group, fill=group), alpha=0.2)


def plot_letsplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    charts: list,
    x_name: str | None = None,
    y_name: str | None = None,
    size: tuple | None = None,
    display_text: bool = False,
    str_x: bool = False,
    str_y: bool = False,
    vjust="outward",
    title: str | None = None,
    x_datetime: bool = False,
    y_datetime: bool = False,
    x_format: str | None = None,
    y_format: str | None = None,
    text_format: str | None = None,
    angle: int = None,
    **chart_param,
):
    df_plot = df.copy()
    if x_name is not None:
        df_plot = df_plot.rename(columns={x: x_name})
        x = x_name
    if y_name is not None:
        df_plot = df_plot.rename(columns={y: y_name})
        y = y_name
    if x_datetime:
        if x_format == "Q":
            df_plot[x] = pd.to_datetime(df_plot[x]).dt.to_period("Q")
        else:
            df_plot[x] = df_plot[x].dt.strftime(x_format)
    if y_datetime:
        if y_format == "Q":
            df_plot[y] = pd.to_datetime(df_plot[y]).dt.to_period("Q")
        else:
            df_plot[y] = df_plot[y].dt.strftime(x_format)
    if str_x:
        df_plot[x] = df_plot[x].astype(str)
    if str_y:
        df_plot[y] = df_plot[y].astype(str)

    lp.LetsPlot.setup_html()
        
    if isinstance(charts, list):
        result = lp.ggplot(df_plot.round(2), lp.aes(x=x, y=y))
        for chart in charts:
            result += chart
    else:
        color = chart_param.get("color", None)
        linetype = chart_param.get("linetype", None)
        result = charts(
            data=df_plot.round(2), 
            mapping=lp.aes(
                x=x, y=y,
            ),
            color=color,
            linetype=linetype
        )
    if display_text:
        result += lp.geom_text(lp.aes(label=y), vjust=vjust, label_format=text_format)
    if size is not None:
        result += lp.ggsize(size[0], size[1])
    if title is not None:
        result += lp.ggtitle(title)
    result += lp.theme(axis_text_x=lp.element_text(angle=angle, hjust=1))
    return result


#########################
# PLOT EXPLORE DATA CHART
#########################
def plot_dist(df, column, sort_value, size=[1200, 400]):
    """
    Tạo biểu đồ phân phối (số lượng và phần trăm) cho một cột dữ liệu.

    Args:
        df (pd.DataFrame): DataFrame đầu vào.
        column (str): Tên cột cần vẽ biểu đồ.
        type_col (str): Loại cột ('num' cho số, 'cat' cho phân loại).

    Returns:
        lets-plot object: Biểu đồ gồm hai biểu đồ thanh - đếm số lượng và phần trăm.
    """
    df_plot = pd.DataFrame(df[column].value_counts()).T[sort_value].T.reset_index()
    
    df_plot['percent'] = 100 * df_plot['count'] / df_plot['count'].sum()
    return lp.gggrid(
        [
            plot_letsplot(
                df=df_plot, x=column, y='count', charts=[letsplot_bar()], 
                str_x=True, display_text=True, vjust=0, angle=15,
            ),
            plot_letsplot(
                df=df_plot, x=column, y='percent', charts=[letsplot_bar()], 
                str_x=True, display_text=True, vjust=0, angle=15, text_format='{}%'
            )
        ]
    ) + lp.ggsize(*size)


def plot_univariate(df, bin_cols={}, limit_unique=5, size=[1200, 400]):
    """
    Vẽ biểu đồ đơn biến (univariate) cho tất cả các cột trong DataFrame.

    - Nếu là cột phân loại (chuỗi): vẽ biểu đồ tần suất.
    - Nếu là cột số: chia bin rồi vẽ biểu đồ.

    Args:
        df (pd.DataFrame): Dữ liệu đầu vào (cột đầu tiên thường là ID, sẽ bỏ qua).
        bin_cols (dict): Từ điển chứa các giá trị phân bin cho các cột số.
    """
    df_plot = df.copy()
    for col in df.columns:
        display(
            Markdown(f"<center><h4 style='font-size:24px'>Distribution of {col}</h4></center>")
        )
        df_plot[col], sort_value = data.process_to_explore(
            df_plot, col=col, bin_cols=bin_cols, limit_unique=limit_unique
        )
        display(
            plot_dist(df=df_plot, column=col, sort_value=sort_value, size=size)
        )


def plot_heatmap(df, title=None, column_names=[None, None], size=[700, 500]):
    df_plot = df.copy()

    df_plot.index.name = None
    df_plot.columns.name = None

    df_plot = df_plot.iloc[::-1]
    
    df_plot = pd.melt(frame=df_plot.reset_index(), id_vars='index', value_vars=df.columns)
    
    return plot_letsplot(
        df=df_plot, 
        x='variable', 
        y='index', 
        charts=[letsplot_heatmap(value='value')], 
        angle=30, 
        title=title,
        x_name=column_names[1],
        y_name=column_names[0]
    ) + lp.ggsize(*size)
    

def plot_crosstab(
    df: pd.DataFrame, columns: list, label_name: str=None, label_elements: str=None,
    angle: int =None, sort_values: list =None, is_transpose=True
):
    dfs_plot = utils.caculate_crosstab(
        df,
        columns=columns,
        sort_values=sort_values,
        is_transpose=is_transpose
    )
    # return dfs_plot
    grid_charts = [
        plot_heatmap(dfs_plot[0], title="(#)", column_names=dfs_plot[4]),
        plot_heatmap(dfs_plot[1], title="(%)", column_names=dfs_plot[4]),
        plot_heatmap(dfs_plot[2], title=f"(%) {dfs_plot[4][0]}", column_names=dfs_plot[4]),
        plot_heatmap(dfs_plot[3], title=f"(%) {dfs_plot[4][1]}", column_names=dfs_plot[4]),
    ]
    
    return lp.gggrid(grid_charts, ncol=2,)


def plot_multivariate(
        df, 
        column, 
        bin_cols={}, 
        is_transpose=True, 
        angle: int =None, 
        reversed_x=False, 
        reversed_y=False, 
        limit_unique=5, 
        size=(1500, 1000)
):
    """
    Vẽ biểu đồ phân tích đa biến giữa một cột chính (`column`) với tất cả các cột còn lại.
    Tùy vào kiểu dữ liệu (chuỗi hoặc số), cột được chia bin hoặc chuyển thành chuỗi để trực quan hóa.

    Args:
        df (pd.DataFrame): Dữ liệu đầu vào.
        column (str): Cột chính cần phân tích.
        bin_cols (dict): Dictionary chứa thông tin bin cho các cột số.

    Returns:
        None: Hiển thị biểu đồ crosstab cho từng cặp biến.
    """
    
    # Duyệt qua tất cả các cột ngoại trừ cột phân tích chính
    for col2 in [col for col in df.columns if col != column]:
        display(
            Markdown(f"<center><h4 style='font-size:24px'>Distribution of {col2}</h4></center>")
        )
        df_plot = df.copy()
        cols = [col2, column]
        sort_values = []  # Danh sách lưu thứ tự phân loại các giá trị trong biểu đồ

        for col in cols:
            df_plot[col], sort_value = data.process_to_explore(
                df=df_plot, col=col, bin_cols=bin_cols, limit_unique=limit_unique
            )
            sort_values.append(sort_value)

        if reversed_x:
            sort_values[0] =  sort_values[0][::-1]

        if reversed_y:
            sort_values[1] =  sort_values[1][::-1]
    
        # return df_plot, sort_values

        # Vẽ biểu đồ phân phối chéo giữa column và col2
        display(
            plot_crosstab(
                df=df_plot,
                columns=cols,
                sort_values=sort_values,
                is_transpose=is_transpose,
                angle=angle
            ) + lp.ggsize(size[0], size[1])
        )
    

def plot_crosstab_by_lable(
    df: pd.DataFrame, 
    columns: list, 
    label_name: str, 
    label_elements=None, 
    sort_values: list=None,
    **kwargs
):
    dfs_plot = utils.caculate_crosstab(
        df,
        columns=columns,
        sort_values=sort_values,
    )
    dfs_plot_good = utils.caculate_crosstab(
        df[df[label_name].eq(label_elements['good'])],
        columns=columns,
        sort_values=sort_values,
    )
    dfs_plot_bad = utils.caculate_crosstab(
        df[df[label_name].eq(label_elements['bad'])],
        columns=columns,
        sort_values=sort_values,
    )

    df_per_good = 100 * dfs_plot_good[0] / dfs_plot[0]
    df_per_bad = 100 * dfs_plot_bad[0] / dfs_plot[0]
    
    return lp.gggrid(
        [
            plot_heatmap(
                df=df_per_good,
                column_names=columns, 
                title="(%) good",
            ),
            plot_heatmap(
                df=df_per_bad,
                column_names=columns, 
                title="(%) bad",
            ),
            plot_heatmap(
                df=dfs_plot_good[0],
                column_names=columns, title="(#) good",
            ),
            plot_heatmap(
                df=dfs_plot_bad[0],
                column_names=columns, title="(#) bad",
            ),
            plot_heatmap(
                df=dfs_plot_good[1],
                column_names=columns, 
                title=f"(%) good all",
            ),
            plot_heatmap(
                df=dfs_plot_bad[1],
                column_names=columns, 
                title=f"(%) bad all"
            ),
            plot_heatmap(
                df=dfs_plot_good[2],
                column_names=columns,
                title=f"(%) good {columns[0]}"
            ),
            plot_heatmap(
                df=dfs_plot_bad[2],
                column_names=columns, title=f"(%) bad {columns[0]}"
            ),
            plot_heatmap(
                df=dfs_plot_good[3],
                column_names=columns, title=f"(%) good {columns[1]}"
            ),
            plot_heatmap(
                df=dfs_plot_bad[3],
                column_names=columns, title=f"(%) bad {columns[1]}"
            ),
        ],
        ncol=2
    )
    

def plot_univariate_by_label(df_bin, size_chart=(1500, 1000), n_limit=None, range_row=None):
    if n_limit is not None:
        columns = df_bin["variable"].unique()[: n_limit]
    elif range_row is not None:
        columns = df_bin["variable"].unique()[range_row[0]: range_row[1]]
    else:
        columns = df_bin["variable"].unique()

    for col in columns:
        df_plot = df_bin[df_bin["variable"] == col].sort_values("index")
        formatted_var = col.replace('_', ' ').title()
        iv_value = df_plot["total_iv"].mean().round(2)
        display(Markdown(f"<center><h4 style='font-size:24px'>Performance by {formatted_var}</h4></center>"))
        display(Markdown(f"<center><h4 style='font-size:24px'>IV: {iv_value}</h4></center>"))
        df_plot["good_distr"] = df_plot["good"] / df_plot["good"].sum()
        df_plot["bad_distr"] = df_plot["bad"] / df_plot["bad"].sum()
        df_plot[["count_distr", "badprob", "bad_distr", "good_distr"]] = (
            100
            * df_plot[["count_distr", "badprob", "bad_distr", "good_distr"]]
        )
        df_plot_total = df_plot[["bin", "count", "badprob", "count_distr"]]
        df_plot_lable = pd.concat(
            [
                df_plot[["bin", "good", "good_distr"]].rename(
                    columns={"good": "count", "good_distr": "count_distr"}
                ).assign(lable="good"),
                df_plot[["bin", "bad", "bad_distr"]].rename(
                    columns={"bad": "count", "bad_distr": "count_distr"}
                ).assign(lable="bad"),
            ]
        )
                
        display(
            lp.gggrid(
                [
                    plot_letsplot(
                        df=df_plot_total, x="bin", y="count", charts=[letsplot_bar()],
                        title="Loan (#)"
                    ) + lp.geom_text(lp.aes(label="count"), vjust="inward", angle=0),
                    plot_letsplot(
                        df=df_plot_total, x="bin", y="count_distr", charts=[letsplot_bar()],
                        title="Loan (%)"
                    ) + lp.geom_text(lp.aes(label="count_distr"), vjust="inward", angle=0, label_format="{}%"),
                    plot_letsplot(
                        df=df_plot_total, x="bin", y="badprob", charts=[letsplot_bar()],
                        title="Bad (%)"
                    ) + lp.geom_text(lp.aes(label="badprob"), vjust="inward", angle=0, label_format="{}%"),
                ],
                ncol=3,
            ) + lp.ggsize(size_chart[0], int(size_chart[1] / 2))
        )

        display(
            lp.gggrid(
                [
                    plot_letsplot(
                        df=df_plot_lable, x="bin", y="count", charts=[letsplot_bar(group="lable")],
                        title="Label (#)"
                    ) + lp.geom_text(
                        lp.aes(label="count", group="lable"), 
                        vjust="inward", position="dodge", angle=0
                    ) + lp.scale_fill_manual(
                        values={
                            "good": "green",
                            "bad": "red"
                        }
                    ),
                    plot_letsplot(
                        df=df_plot_lable, x="bin", y="count_distr", charts=[letsplot_bar(group="lable")],
                        title="Label (%)"
                    ) + lp.geom_text(
                        lp.aes(label="count_distr", group="lable"), 
                        vjust="inward", position="dodge", label_format="{}%", angle=0
                    ) + lp.scale_fill_manual(
                        values={
                            "good": "green",
                            "bad": "red"
                        }
                    ),
                ],
                ncol=2
            ) + lp.ggsize(size_chart[0], int(size_chart[1] / 2))
        )


##############################
# PLOT METRIC TO EVALUTE MODEL
##############################
def plot_ROC(y_true, y_probPred):
    fpr, tpr, thresholds = M.roc_curve(y_true, y_probPred)
    df_roc = pd.DataFrame(
        {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
        }
    )
    df_roc["Note"] = "ROC cuver"
    df_baseline = pd.DataFrame({"fpr": [0, 1], "tpr": [0, 1]})
    df_baseline["Note"] = "baseline"
    df_plot = pd.concat([df_roc, df_baseline])
    df_plot["Note"] = df_plot["Note"].astype("category")
    
    return (
        plot_letsplot(
            df=df_plot,
            x="fpr",
            y="tpr",
            charts=[letsplot_line(group="Note")]
        ) 
        + lp.scale_color_manual(
            {
                "ROC cuver": "blue",
                "baseline": "black",
            }
        ) 
        + lp.scale_linetype_manual(
            {
                "ROC cuver": "solid",
                "baseline": "dashed",
            }
        )
        + lp.ggtitle(f"ROC: {M.auc(df_roc["fpr"], df_roc["tpr"]).round(4)}")
    )


def plot_KS(y_true, y_probPred):
    pos_score = np.sort(y_probPred[y_true[y_true.eq(1)].index])
    neg_score = np.sort(y_probPred[y_true[y_true.eq(0)].index])

    df_pos = pd.DataFrame(
        {
            "p": pos_score,
            "cumulative": np.linspace(0, 1, len(pos_score)),
        }
    )
    
    df_neg = pd.DataFrame(
        {
            "p": neg_score,
            "cumulative": np.linspace(0, 1, len(neg_score)),
        }
    )
    
    df_pos["Note"] = "positive"
    df_neg["Note"] = "negative"
    
    df_baseline = pd.merge(df_pos, df_neg, on="p", how="outer", suffixes=("_pos", "_neg")).sort_values("p").interpolate()
    df_baseline["KS"] = (df_baseline["cumulative_pos"] - df_baseline["cumulative_neg"]).abs()
    df_baseline = df_baseline[df_baseline["KS"].eq(df_baseline["KS"].max())].iloc[[0]]
    df_baseline = pd.DataFrame(
        {
            "p": [df_baseline["p"].iloc[0], df_baseline["p"].iloc[0]],
            "cumulative": [df_baseline["cumulative_pos"].iloc[0], df_baseline["cumulative_neg"].iloc[0]]
        }
    )
    df_baseline["Note"] = "baseline"
    
    df_plot = pd.concat([df_pos, df_neg, df_baseline])
    
    return (
        plot_letsplot(
            df=df_plot,
            x="p",
            y="cumulative",
            charts=[letsplot_line(group="Note")]
        ) 
        + lp.scale_linetype_manual(
            {
                "positive": "solid",
                "negative": "solid",
                "baseline": "dashed",
            }
        )
        + lp.scale_color_manual(
            {
                "positive": "blue",
                "negative": "red",
                "baseline": "black",
            }
        )
        + lp.ggtitle(
            f"KS: {(df_baseline["cumulative"].iloc[1] - df_baseline["cumulative"].iloc[0]).round(4)}, p={df_baseline["p"].iloc[0].round(4)}"
        )
    )


def plot_lift(y_train, y_probPred):
    df_lift_pos = metric.caculate_lift(y_train, y_probPred, n_bins=10)
    df_lift_pos["Note"] = "positive"
    df_lift_neg = metric.caculate_lift(y_train, y_probPred, type_class=0, n_bins=10)
    df_lift_neg["Note"] = "negative"
    df_plot = pd.concat(
        [
            df_lift_pos, df_lift_neg,
            pd.DataFrame(
                {
                    "random_rate": [df_lift_pos["random_rate"].iloc[0], 1],
                    "lift": [1, 1],
                    "Note": ["baseline", "baseline"],
                }
            )
        ]
    )
    return (
        plot_letsplot(
            df=df_plot,
            x="random_rate",
            y="lift",
            charts=[letsplot_line(group="Note")]
        ) 
        + lp.scale_linetype_manual(
            {
                "positive": "solid",
                "negative": "solid",
                "baseline": "dashed",
            }
        )
        + lp.scale_color_manual(
            {
                "positive": "blue",
                "negative": "red",
                "baseline": "black",
            }
        ) 
    )


def plot_gain(y_train, y_probPred):
    df_gain = metric.caculate_gain(y_train, y_probPred)
    df_gain["Note"] = "positive"
    
    df_baseline = pd.DataFrame(
        {
            "percent_sample": [0, 1],
            "gain": [0, 1]
        }
    )
    df_baseline["Note"] = "baseline"
    
    df_plot = pd.concat([df_gain, df_baseline])
    return (
        plot_letsplot(
            df=df_plot,
            x="percent_sample",
            y="gain",
            charts=[letsplot_line(group="Note")]
        ) 
        + lp.scale_linetype_manual(
            {
                "positive": "solid",
                "baseline": "dashed",
            }
        )
        + lp.scale_color_manual(
            {
                "positive": "blue",
                "baseline": "black",
            }
        ) 
    )


def plot_evaluted_classification_metric(y_true, y_probPred):
    display(
        lp.gggrid(
            [
                plot_ROC(y_true, y_probPred),
                plot_KS(y_true, y_probPred),
                plot_lift(y_true, y_probPred),
                plot_gain(y_true, y_probPred),
            ],
            ncol=2
        )   
    )