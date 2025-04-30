"""
Written by: Lars Daniel Johansson Niño
Created date: 09/08/2024
Purpose: Prepare data for analysis 2. Take a lower amount of points, current levels arent managable for R. 
Notes: 
"""
import polars as pl
import matplotlib.pyplot as plt 
import os

file_p = (os.path.realpath(__file__)).replace('GatherData3d1.py', '') #Gets current file path.
os.chdir(file_p) #Changes working directory to current file´s path. 
cwd = os.getcwd() #Reads the new working directory. 

curr_data = pl.read_csv(  'data_flights_egll_esgg1.csv', has_header=True) #Reads current data.
k = 50
n_points = []
print(curr_data)
cols = [pl.Series('id', [], dtype = pl.String),pl.Series('t', [], pl.Float64), pl.Series('x_lat', [],  dtype= pl.Float64), pl.Series('y_lon', [], dtype= pl.Float64), pl.Series('n', [], dtype = pl.Int64)] #Creates columns for final data. 
final_data = pl.DataFrame(cols) #Creates data frame to store final data. 

flights = curr_data['n'].unique().to_list()

for f in flights:
    cf = curr_data.filter(pl.col('n') == f) #Selects those values with number n. 
    cf = cf.with_columns(pl.Series('keep',[ (i % k) == 0 for i in range(0, len(cf))] )) #Keeps all index values multiples of k.
    cf = cf.filter(pl.col('keep') == True) #Selects the index values multiples of k.
    cf = cf.drop('keep') 
    n_points.append(len(cf))
    final_data.extend(cf)
print(final_data)
plt.hist(n_points) #Shows a histogram 
plt.show()
final_data.write_csv('data_flights_egll_esgg2.csv')
