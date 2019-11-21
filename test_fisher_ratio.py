from fisher_ratio import fisher_ratio
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


aux_array = np.concatenate([np.arange(0, 2), np.arange(100, 103), np.arange(10, 14)])
aux_groups = ["a"]*2 + ["b"]*3 + ["c"]*4

df = pd.DataFrame({"dim1": 1.0*aux_array,
                   "dim2": 1.0*aux_array*2 + 250,
                   "dim3": 1.0*aux_array*3 + 500,
                   "group": aux_groups})

print("\nunscaled data\n", df)

df = df.drop("group", axis="columns")  # Drop group column before scaling

scaler = StandardScaler()
scaler.fit(df)

# Scaled data and restore group column
new_df = pd.DataFrame(scaler.transform(df), columns=df.columns)
new_df.loc[:, "group"] = aux_groups

print("\nscaled data\n", new_df)

print("\nfisher ratio:", fisher_ratio(new_df))
