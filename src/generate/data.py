import pandas as pd
import numpy as np
import datetime

path = "C:\\Users\\lokesh-g\\Downloads\\IU"

csv_filename1 = f"{path}\\indiana_projections.csv"
csv_filename2 = f"{path}\\indiana_reports.csv"
image_base_path = f"{path}\\images\\images_normalized"

output_csv_filename = f"{path}\\indiana_merged_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv" # only first 1000 rows
# absolute path to the output csv file
output_filepath = f"C:\\Users\\lokesh-g\\Downloads\\IU\\indiana_merged.csv"

def merge_csv_files_andupdate_image_path():
    # merge the two CSV files  on="uid", how="inner"
    df1 = pd.read_csv(csv_filename1)
    df2 = pd.read_csv(csv_filename2)

    merged_df = pd.merge(df1, df2, on="uid", how="inner")
    # update the "image_path" column to include the base path
    merged_df["image_path"] = merged_df["filename"].apply(lambda x: f"{image_base_path}/{x}") 


    # save the merged dataframe to a new CSV file only first 1000 rows
    merged_df.to_csv(output_csv_filename, index=False)

## read the merged csv file and print the first 5 rows
def read_merged_csv():
    merged_df = pd.read_csv(output_csv_filename)
    print("Length of merged dataframe:", len(merged_df))
    print(merged_df.head(5)) 

if __name__ == "__main__":
    merge_csv_files_andupdate_image_path()
    read_merged_csv()

    print