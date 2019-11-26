from fisher_ratio import fisher_ratio, scale_xy
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
print("\nfisher ratio unscaled:", fisher_ratio(df))

new_df = scale_xy(df, label_y="group")

print("\nscaled data\n", new_df)
print("\nfisher ratio scaled:", fisher_ratio(new_df))


aux_array2 = np.concatenate([np.arange(0, 3), np.arange(100, 103), np.arange(10, 13)])
aux_groups2 = ["a"]*3 + ["b"]*3 + ["c"]*3

df2 = pd.DataFrame({"dim1": 1.0*aux_array2,
                    "dim2": 1.0*aux_array2,
                    "group": aux_groups2})

print("\nunscaled data\n", df2)
print("\nfisher ratio unscaled:", fisher_ratio(df2))

new_df2 = scale_xy(df2, label_y="group")

print("\nscaled data\n", new_df2)
print("\nfisher ratio scaled:", fisher_ratio(new_df2))


aux_array3 = np.concatenate([np.arange(0, 3), np.arange(0, 3), np.arange(0, 3)])
aux_groups3 = ["a"]*3 + ["b"]*3 + ["c"]*3


df3 = pd.DataFrame({"dim1": 1.0*aux_array3,
                    "dim2": 1.0*aux_array3,
                    "group": aux_groups3})

print("\nunscaled data\n", df3)
print("\nfisher ratio unscaled:", fisher_ratio(df3))

new_df3 = scale_xy(df3, label_y="group")

print("\nscaled data\n", new_df3)
print("\nfisher ratio scaled:", fisher_ratio(new_df3))
