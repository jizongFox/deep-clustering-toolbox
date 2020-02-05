import pandas as pd


def rename_df_columns(dataframe: pd.DataFrame, name: str):
    dataframe.columns = list(map(lambda x: name + "_" + x, dataframe.columns))
    return dataframe
