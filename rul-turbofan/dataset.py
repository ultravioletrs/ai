import pandas as pd

column_names = [
    "unit_number", "time_in_cycles", "operational_setting_1", "operational_setting_2", "operational_setting_3"
] + [f"sensor_measurement_{i}" for i in range(1, 27)]

df_train = pd.read_csv("train_FD001.txt", sep=r'\s+', header=None, names=column_names)

print(df_train.head(20))
