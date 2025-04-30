import os
scrpt_path = os.path.abspath(__file__) #Gets absolute path of the current script. 
scrpt_dir = os.path.dirname(scrpt_path) # Get the directory name of the current script
os.chdir(scrpt_dir) #Chenges working direct  ory to the python file´s path. 

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pandas as pd
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


open_file_name =  'rfpca_euclidean_flights_egll_esgg1d2.csv' 
#open_file_name =  'rfpca_spherical_flights_egll_esgg1d2.csv'
#open_file_name = 'rfpca_work_flights_egll_esgg1d2.csv' 
#open_file_name =  'rfpca_flights_egll_esgg1d2.csv'  
#open_file_name = 'fpca_org_flights_egll_esgg1d2.csv'

subs = 's0'
color_col = 'cs'

K =12

sphere_data = pd.read_csv(open_file_name)

fig, ax = plt.subplots(4, 3, subplot_kw={'projection': '3d'}, figsize=(15, 9))
ax = ax.flatten()


cmap = cm.get_cmap("tab10", K) 

#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')

#P = np.zeros((450,3, K))

for i in range(K):
    phi = sphere_data[sphere_data['lbl'] == ('phi ' + str(i+1))]
    color = cmap(i)  # Assign a unique color

    
    x = phi['x'].to_numpy()
    y = phi['y'].to_numpy()
    z = phi['z'].to_numpy()

    norm = np.sqrt(x**2 + y**2 + z**2)
    print(f"phi {i+1}  mean of norms: {np.mean(norm)} \nvariance of norm {np.var(norm)}\n")

    #P[:, 0, i] = x
    #P[:,1,i] = y 
    #P[:, 2, i] = z
    ax[i].plot(x,y,z,  linewidth = 2.2, alpha = 0.9, color = color,  label=f"EF {i+1}")
    ax[i].legend()

r = 2
print(f"With respect to {r}")
#for i in range(0,K):
    #L = P[:,:, i] - P[:, :, r]
    #l = np.sum(L)/L.shape[0]*L.shape[1]
    #print(f"For {i} mean entry difference {l}")






mu = sphere_data[sphere_data['lbl'] == 'mu_est']
mu_x = mu['x'].to_numpy()
mu_y = mu['y'].to_numpy()
mu_z = mu['z'].to_numpy()


phi_mu = np.arccos(mu_z)
theta_mu = np.arcsin(mu_y/np.sin(phi_mu) )

phi_min = np.min(phi_mu)
phi_max = np.max(phi_mu)

theta_min = np.min(theta_mu)
theta_max = np.max(theta_mu)
l = 0.01


figS = plt.figure(figsize=(8, 6))
axS = figS.add_subplot(111, projection='3d')


# Plot the unit sphere
#uS = np.linspace(0, 2 * np.pi, 100)
#vS = np.linspace(0, np.pi, 100)

uS = np.linspace(theta_min-l, theta_max+l, 100)
vS = np.linspace(phi_min-l, phi_max+l, 100)

xS = np.outer(np.cos(uS), np.sin(vS))
yS = np.outer(np.sin(uS), np.sin(vS))
zS = np.outer(np.ones_like(uS), np.cos(vS))

axS.plot_surface(xS, yS, zS, color='lightgray', alpha=0.3, linewidth=0)

axS.plot(mu_x, mu_y, mu_z, color = 'red')

#ax.plot(mu_x ,mu_y, mu_z, color = "red", linewidth = 2)

#--Create figure to plot data--------------------------------------------------------------------------------------------------------------

flight_data_file =  'smoothed_less_p_data_flights_egll_esgg1d2.csv'# 'egll_esgg_clustered_0vals.csv'  #'less_p_data_flights_egll_esgg1d2.csv'

data = pl.read_csv(  flight_data_file, has_header= True)
flight_n= data['n'].unique().to_list()

fig1, ax1 = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(15, 9))
ax1 = ax1.flatten()

ax1[1].plot(mu_x ,mu_y, mu_z, color = "red", linewidth = 2.5, label = 'Mu')
ax1[1].legend()


unique_cs = data[color_col].unique().to_list() #Creates list of unique callsigns for flights. 
cmap = plt.cm.viridis  # You can choose a different colormap, e.g., 'plasma', 'cividis', etc.
colors = cmap(np.linspace(0, 1, len(unique_cs))) #Creates a unique color for each callsign. 

for f in flight_n: #[0:1]:
    print(f"In flight {f}")
    cf = data.filter(pl.col('n') == f) #Selects those values with number n. 
    cf_call_s = cf.get_column(color_col)[0]
    cf_color = colors[unique_cs.index(cf_call_s)]
    x_cf = cf[f'x_{subs}'].to_list()
    y_cf = cf[f'y_{subs}'].to_list()
    z_cf = cf[f'z_{subs}'].to_list()
    axS.plot(x_cf, y_cf, z_cf, color = cf_color, alpha = 0.4)

    t = cf['t'].to_list()
    #print(f"Number of points {len(x_cf)}")
    ax1[0].plot(x_cf, y_cf, z_cf, color = cf_color)
ax1[0].legend()


plt.tight_layout()
plt.show()





plt.show()


