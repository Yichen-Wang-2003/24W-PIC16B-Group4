import pandas as pd
import math

def target_setting(df):
    """
    vectorized operation to calculate the target value based on formula
    """
    df['-rt'] = -0.04*(df['[EXPIRE_UNIX]'] - df['[QUOTE_UNIXTIME]'])/(3600*365*24)  # unix time is based on seconds
    df['price_diff'] = df['[STRIKE]'] - df['Adj Close']
    df['exp(-rt)'] = df['-rt'].apply(lambda x: math.exp(x))
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]   
    df['discounted_price'] = df['price_diff'] * df['exp(-rt)']
    return df


