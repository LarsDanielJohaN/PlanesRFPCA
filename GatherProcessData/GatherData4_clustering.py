"""
Created by: Lars Daniel Johansson Niño
Last edited: 07/02/2025
Purpose: Cluster functional data. 


"""

from sklearn.cluster import KMeans
import numpy as np
import polars as pl
import matplotlib.pyplot as plt 
import os

file_p = (os.path.realpath(__file__)).replace('GatherData4_clustering.py', '') #Gets current file path.
os.chdir(file_p) #Changes working directory to current file´s path. 
cwd = os.getcwd() #Reads the new working directory. 

file_to_open = 'less_p_data_flights_egll_esgg1d2.csv' #'data_flights_egll_esgg1.csv'
curr_data = pl.read_csv(  file_to_open, has_header=True) #Reads current data.



print(curr_data)
exit()
save_file = 'egll_esgg_clustered_0vals.csv'

subs = 'unit_0'

flights = curr_data['n'].unique().to_list()
n_obs = 100
k = 3
no_cl = 4
flight_data = np.zeros(shape = (len(flights), k*n_obs))


for f in flights:
    cf = curr_data.filter(pl.col('n') == f) #Selects those values with number n. 

    cf_x = cf[f'x_{subs}'].to_numpy()
    cf_y = cf[f'y_{subs}'].to_numpy()
    cf_z = cf[f'z_{subs}'].to_numpy()

    cf_d = np.hstack( (cf_z, cf_x)  )
    cf_d = np.hstack( (cf_d, cf_y)  )

    flight_data[f-1, :] = cf_d[0:k*n_obs] 
    if k*n_obs != len(cf_d):
        print(f"Flight without {k}00  ", f)

kmeans = KMeans(n_clusters=no_cl, init = 'k-means++',random_state=0, n_init="auto").fit(flight_data)

aux_lbs = pl.DataFrame( {'n':flights, 'cluster':kmeans.labels_}  )
curr_data = curr_data.join(aux_lbs, on = 'n', how = 'left')

curr_data.write_csv(save_file)

print(kmeans.labels_)
print()
print(curr_data)


