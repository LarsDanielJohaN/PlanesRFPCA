"""
Written by: Lars Daniel Johansson Niño
Created date: 09/08/2024
Purpose: Prepare data for analysis 2. Take a lower amount of points, current levels arent managable for R. 
Notes: 
"""
import polars as pl
import matplotlib.pyplot as plt 
import numpy as np
import os

file_p = (os.path.realpath(__file__)).replace('GatherData3.py', '') #Gets current file path.
os.chdir(file_p) #Changes working directory to current file´s path. 
cwd = os.getcwd() #Reads the new working directory. 

file_to_open = 'data_flights_egll_esgg1d2.csv' #'data_flights_egll_esgg1.csv'
curr_data = pl.read_csv(  file_to_open, has_header=True) #Reads current data.
k = 10
n_points = []
curr_data_cols = curr_data.columns
curr_data_types = curr_data.dtypes
#cols = [pl.Series('id', [], dtype = pl.String),pl.Series('t', [], pl.Float64), pl.Series('x_lat', [],  dtype= pl.Float64), pl.Series('y_lon', [], dtype= pl.Float64),    pl.Series('z_alt', [], dtype= pl.Float64),   pl.Series('call_s', [], dtype = pl.String),   pl.Series('n', [], dtype = pl.Int64),  pl.Series('x_unit', [], pl.Float64), pl.Series('y_unit', [], pl.Float64), pl.Series('z_unit', [], pl.Float64)] #Creates columns for final data. 
#cols = [pl.Series('id', [], dtype = pl.String),pl.Series('t', [], pl.Float64), pl.Series('x_lat', [],  dtype= pl.Float64), pl.Series('y_lon', [], dtype= pl.Float64), pl.Series('z_alt', [], dtype = pl.Float64), pl.Series('call_s', [], dtype = pl.String), pl.Series('n', [], dtype = pl.Int64)] #Creates columns for final data. 
cols = [pl.Series( curr_data_cols[i], dtype = curr_data_types[i]) for i in range(len(curr_data_cols))]
final_data = pl.DataFrame(cols) #Creates data frame to store final data. 


flights = curr_data['n'].unique().to_list()
reps = curr_data['n'].value_counts()


to_keep_ish =700

reps = reps.with_columns( (pl.col('count')/to_keep_ish  ).cast(pl.Int32).alias('to_keep_ish') )
p =reps['to_keep_ish'].min()
#reps = reps.with_columns(  pl.col('to_keep_ish').cast(pl.Int32))

for f in flights:
    cf = curr_data.filter(pl.col('n') == f) #Selects those values with number n. 

    p = int(np.round(len(cf)/to_keep_ish))

    if p == 0:
        p = len(cf)

    cf = cf.with_columns(pl.Series('keep',[ (i%p) == 0 for i in range(0, len(cf))] )) #Keeps all index values multiples of k.
    cf = cf.filter(pl.col('keep') == True) #Selects the index values multiples of k.
    cf = cf.drop('keep') 

    if len(cf)%2 == 1:
        cf = cf.slice(1, cf.height - 1)
    print("Resulting shape:  ", cf.shape )
    n_points.append(len(cf))
    final_data.extend(cf)


    print(".-......................................................")


print(final_data)
plt.hist(n_points) #Shows a histogram 
plt.show()
save_file = 'less_p_'+file_to_open
final_data.write_csv(save_file)
