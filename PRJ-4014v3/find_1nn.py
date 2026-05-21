from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale, RobustScaler, LabelEncoder
from sklearn.metrics.pairwise import kernel_metrics

def Scaler(*x):
    for idx, data in enumerate(x):
        if idx == 0:
            xm_slr = RobustScaler().fit(data)
            xm_data = xm_slr.transform(data)
            if len(x) == 1:
                return xm_slr, xm_data
        elif idx == 1:
            xa_slr = RobustScaler().fit(data)
            xa_data = xa_slr.transform(data)
            return xm_slr, xm_data, xa_slr, xa_data


pre_record = pd.read_excel('RecordSummary input for each case.xlsx')

pre_record = pre_record[['Depth', 'Epidst', 'GRDACC', 'APKV2', 'VPK', 'DPK', 'SA03', 'SA1', 'SA3', 'Stracc']]

train = pre_record.iloc[:000] # Specify a row of interest
test = pre_record.iloc[000]

slr, train = Scaler(train)
test = slr.transform(test.values.reshape(1,-1))

nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric='chebyshev').fit(train)
distances, indices = nbrs.kneighbors(test)