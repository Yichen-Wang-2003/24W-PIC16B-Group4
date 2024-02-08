import pandas as pd

def prep_txt(path):
    """
    Input a path with option txt data and out put a dataframe
    """
    df = pd.read_table(path)    # read the txt file
    result_df = df[df.columns[0]].str.split(',', expand=True)   # split the values
    column_names = df.columns[0].split(',')     # split the column names
    result_df.columns = column_names    # renaming column names
    return result_df