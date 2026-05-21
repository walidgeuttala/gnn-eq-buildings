import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

folder = 'rawdata path/'
filename = 'rawdata filename'
time_unit = 0.001
save_folder = 'result saving path/'

colspecs = [(0,10),(10,20),(20,30),(30,40),(40,50),(50,60),(60,70),(70,80)]

df_acc = pd.read_fwf(folder + filename + '.V2_acc', header=None, skiprows=1, colspecs=colspecs)
df_acc = df_acc.iloc[250:,:]
acc_array = df_acc.values.flatten().reshape(-1,1) *10 # Unit conversion from cm/s^2 to mm/s^2

df_disp = pd.read_fwf(folder + filename + '.V2_disp', header=None, skiprows=1, colspecs=colspecs)
df_disp = df_disp.iloc[250:,:]
disp_array = df_disp.values.flatten().reshape(-1,1) *10 # Unit conversion from cm to mm

time_step = np.array([(i+1)*time_unit for i in range(len(acc_array))]).reshape(-1,1)

df_all = pd.DataFrame(np.concatenate((time_step, acc_array, disp_array), axis=1), columns=['time','acc','disp'])
df_all.to_excel(save_folder + filename + '.xlsx', header=True, index=False)
