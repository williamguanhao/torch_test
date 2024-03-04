import os
import pandas as pd
def read_to_df(file, root):
    file_path = root + '/' + file
    pass

def data_cleaning(df):
    cleaned_df = df
    return cleaned_df

def data_reconstruct(df):
    reconstructed_df = df
    return reconstructed_df

def consolidate_df(df1, df2):
    return pd.concat([df1, df2])

def get_useful_dta(dir_path):
    out_df = pd.DataFrame()
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directory and {len(filenames)} files in {dirpath}")
        print("Consolidating data...")
        df = read_to_df(filenames, dirpath+'/'+dirnames)
        cleaned_df = data_cleaning(df)
        reconstructed_df = data_reconstruct(cleaned_df)
        out_df = consolidate_df(out_df, reconstructed_df)
    out_df.to_csv('useful_data.csv')
