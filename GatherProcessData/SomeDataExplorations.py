"""
Written by: Lars Daniel Johansson Niño
Created date: 13/08/2024
Purpose: Make some data visualizations. 
Notes: 
"""
print("Hello SomeDataexplorations!!!")

import polars as pl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from numpy.random import multivariate_normal
import numpy as np
import os

file_p = (os.path.realpath(__file__)).replace('SomeDataExplorations.py', '') #Gets current file path.
os.chdir(file_p) #Changes working directory to current file´s path. 
cwd = os.getcwd() #Reads the new working directory. 


open_folder = 'data2'
open_file_name =  'data_flights_egll_esgg1d2.csv'  #'less_p_data_flights_egll_esgg1d2.csv'
open_file_name = 'smoothed_less_p_data_flights_egll_esgg1d2.csv'
#open_file_name = 'less_p_data_flights_egll_esgg1d2.csv'
data = pl.read_csv(  open_file_name, has_header= True)
flight_n= data['n'].unique().to_list()

#data = data.rename( {'x_lat':'x_org', 'y_lon':'y_org', 'z_alt':'z_org'}) #Renames latitude and longitude colums to x_lat and y_lon respectivelly. 




#--Create figure to plot data--------------------------------------------------------------------------------------------------------------
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

fig_x = plt.figure()
ax_x = fig_x.add_subplot()

fig_y = plt.figure()
ax_y = fig_y.add_subplot()

fig_z = plt.figure()
ax_z = fig_z.add_subplot()


#------------------------------------------------------------------------------------------------------------------------------------------

subs =  'r0'  #'unit'
color_col =  'cs' #'call_s' #'cs' 
unique_cs = data[color_col].unique().to_list() #Creates list of unique callsigns for flights. 
cmap = plt.cm.viridis  # You can choose a different colormap, e.g., 'plasma', 'cividis', etc.
colors = cmap(np.linspace(0, 1, len(unique_cs))) #Creates a unique color for each callsign. 

print(unique_cs)

for f in flight_n: #[0:1]:
    print(f"In flight {f}")
    cf = data.filter(pl.col('n') == f) #Selects those values with number n. 
    print(f"Shape: {cf.shape}")
    cf_call_s = cf.get_column(color_col)[0]
    cf_color = colors[unique_cs.index(cf_call_s)]
    x_cf = cf[f'x_{subs}'].to_list()
    y_cf = cf[f'y_{subs}'].to_list()
    z_cf = cf[f'z_{subs}'].to_list()
    t = cf['t'].to_list()


    #fix1, ax1 = plt.subplots(nrows = 1, ncols = 2)
    #ax1[0].hist(z_cf)
    #ax1[1].plot(t, z_cf)
    #plt.show()

    #print(f"Number of points {len(x_cf)}")
    ax.plot(x_cf, y_cf, z_cf, color = cf_color)
    ax_x.plot(t, x_cf)
    ax_y.plot(t, y_cf)
    ax_z.plot(t, z_cf)


ax.set_box_aspect([1, 1, 1])  # Equal scaling
ax_x.set_title("x")
ax_y.set_title("y")
ax_z.set_title("z")

plt.show()