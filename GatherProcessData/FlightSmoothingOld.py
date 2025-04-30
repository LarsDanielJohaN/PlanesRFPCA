"""
Written by: Lars Daniel Johansson Niño
Created date: 09/08/2024
Purpose: Create splines for functional data and returns their images under an equally spaced on [0,1]
Notes: 
"""
print("Hello GatherData3d2!")

import polars as pl
import sympy as sp
from sympy.utilities.lambdify import implemented_function
import matplotlib.pyplot as plt 
import numpy as np
import os

file_p = (os.path.realpath(__file__)).replace('GatherData3d2.py', '') #Gets current file path.
os.chdir(file_p) #Changes working directory to current file´s path. 
cwd = os.getcwd() #Reads the new working directory. 
f, axs = plt.subplots(2,3)
f_aux, axs_aux = plt.subplots(1,1)
f_alt, axs_alt = plt.subplots(1,1)

axs_aux.set_title('Vuelos de Londres a Gotemburgo')
axs_aux.set_xlabel('Latitud X')
axs_aux.set_ylabel('Longitud Y')

axs[0,0].set_title('Xi(t) observado')
axs[0,0].set_xlabel('t')
axs[0,0].set_ylabel('Latitud X')


axs[0,1].set_title('Yi(t) observado')
axs[0,1].set_xlabel('t')
axs[0,1].set_ylabel('Longitud Y')


axs[0,2].set_title('(X(t), Y(t)) observado')
axs[0,2].set_xlabel('Latitud X')
axs[0,2].set_ylabel('Longitud Y')



axs[1,0].set_title('Xi(t) aproximado')
axs[1,0].set_xlabel('t')
axs[1,0].set_ylabel('Latitud X')


axs[1,1].set_title('Yi(t) aproximado')
axs[1,1].set_xlabel('t')
axs[1,1].set_ylabel('Longitud Y')


axs[1,2].set_title('(X(t), Y(t)) aproximado')
axs[1,2].set_xlabel('Latitud X')
axs[1,2].set_ylabel('Longitud Y')


    
def get_x_mat(tv, p_d, knots): #Obtains a desired design matrix for OLS optimization problem (created as method to allow flexibility in the future). 
    x_pol = np.array([tv**i for i in range(0, p_d+1)]) # Creates a matrix with each row equal to t**i, thus the name x_pol. 
    x = x_pol
    return np.transpose(x) #Returns the transpose of the created x matrix. 

def get_coef_vec(X, Y): #Solves by first facorizing X = QR, performing the product QtY = [c d] and finally solving RB = c to obtain the coefficient matrix. 
                        # c is equal to the first col values of the QtY vector.
    row, col = X.shape  #Gets dimensions of the design matrix. 
    Q,R = np.linalg.qr(X, 'complete') #Performs QR factorization on the X matrix and sotres it in corresponding variables. 
    Qt = np.transpose(Q) #Obtains tranpose of the Q matrix. 
    QtY = np.dot(Qt, Y) #Performs QtY product. 
    beta = np.linalg.solve(R[0:col,:], QtY[0:col]) #Solves RB = c for B (beta). 
    return beta

def get_apprx(beta, x, p_d):
    xv = np.array([x**i for i in range(0, p_d + 1 )])
    return np.dot(beta, xv)

def get_flight_color(call_s, unq_cs, cs_clrs): #Returns color for unique callsign. 
    clr_id = unq_cs.index(call_s) #Gets possition value
    return cs_clrs[clr_id] #Returns corresponding color. 

def write_fin_df(x_lat, y_lon, id, t_v):
    x_lat = np.matrix(x_lat) #Converts x_lat list to a numpy matrix, convenient to define data frame. 
    y_lon = np.matrix(y_lon) #Converts y_lon  list to a numpy matrix, convenient to define data frame. 
    xr, xc = x_lat.shape #Gets dimensions of the x_lat matrix. 
    yr, yc = y_lon.shape #Gets dimensions of the y_lon matrix. 

    cols_x = {f'p{i+1}': x_lat[:,i].A1.tolist() for i in range(0, xc)} #Takes x_lat matrix and creates a dictionary per matrix column (corresponding to all latitude vals on the i-th point)
    cols_y = {f'p{i+1}': y_lon[:,i].A1.tolist() for i in range(0, yc)} #Takes y_lon matrix and creates a dictionary per matrix column (corresponding to all longitude vals on the i-th point)
    cols_x['id'] = id.copy() #Creates id element ni cols_x dictionary. Order must be preserved so that id matches its recorded values. 
    cols_y['id'] = id.copy() #Creates id element ni cols_y dictionary. Order must be preserved so that id matches its recorded values.

    cols_x['id'].append('dom') #Adds an dom id element correspondant to the domain grid values. 
    cols_y['id'].append('dom') #Adds an dom id element correspondant to the domain grid values. 

    for i in range(0, xc): #Adds a row with domain values both for latitude and longitude. In general, xc = yc = len(t_v) is true. 
        cols_x[f'p{i+1}'].append(t_v[i]) #Adds i-th element of the new domain grid to cols_x. 
        cols_y[f'p{i+1}'].append(t_v[i]) #Adds i-th element of the new domain grid to cols_y. 

    return [pl.DataFrame(cols_x), pl.DataFrame(cols_y)] #Returns a length two list with resulting polars data frames with latitude and longitude fits. 

x_lat_file = 'x_lat_data_flights_egll_esgg3d2.csv' #Defines file to store the x_lat values. 
y_lon_file = 'y_lon_data_flights_egll_esgg3d2.csv' #Decines the file to store the y_lon values. 
alt_file = 'alt_data_flights_egll_esgg3d2.csv' #Defines file to store the altitude values. 

n_points_t = 100 #Defines the number of points that the new evaluation grid will contain. 
t_grid =  np.arange(0,1, 1/n_points_t) #Creates the evaluation grid for the approximations, common to all flights. 
p_d = 20 #Defines the degree of the polinomial to fit for each flight (the same for all of them)
err_x = [] #Creates list to store the sum of squared errors for x. 
err_y = [] #Creates list to store the sum of squared errors for y. 

x_vals = [] #Creates list to store the x approximations evaluated on the grid. 
y_vals = [] #Creates list to store the y approximations evaluated on the grid. 
f_ids = [] #Creates a list to store the flight id´s. 

t = sp.symbols('t')
curr_data = pl.read_csv(  'data_flights_egll_esgg1d2.csv', has_header=True) #Reads current data.
n_v = curr_data['n'].unique().to_list() #Gets a list with the unique flight numbers. 
unq_cs =curr_data['call_s'].unique().to_list()
cmap = plt.cm.get_cmap('hsv', len(unq_cs))
cs_clrs = [cmap(i) for i in range(len(unq_cs))]


#The following section iterates through the flights and finds an approximation to them through a polinomial of degree p_d. 
#The fit is done by solving an optimization problem through OLS (XB = Y) by making a QR decomposition of the design matrix X. 
for n in n_v: 
    try:
        print("Going for flight ",n)
        curr_f = curr_data.filter(pl.col('n') == n) #Selects only those datapoints corresponding to flight n. 

        f_ids.append(curr_f.row(0)[0]) #Adds the n-th´s flight id to the f_ids list. 
        t_v = np.array(curr_f['t'].to_list()) #Takes evaluation points for the n-th flight and converts them to an np array. 
        x_lat = np.array(curr_f['x_lat'].to_list()) #Takes the latitude points for the n-th flight and converts them to an np array. 
        y_lon = np.array(curr_f['y_lon'].to_list()) #Takes the longitude points for the n-th flight and converts them to an np array. 

        axs[0,0].plot(t_v, x_lat) #Plots original values of latitude (x) vs the recorded points which python unites.  
        axs[0,1].plot(t_v, y_lon) #Plots original values of longitude (y) vs the recorded points which python unitess. 
        axs[0,2].plot(x_lat, y_lon) #Plots the original latitude (x) vs longitude (y) values which python unites as two dim coordinates. 
        axs_aux.plot(x_lat, y_lon) #Plots the original latitude (x) vs longitude (y) values which python unites as two dim coordinates. 
        axs_alt.plot(t_v, curr_f['alt'], get_flight_color(curr_f.row(0)[5], unq_cs, cs_clrs)) #Plots the alitude of time vrs altitude. 


        des_mat = get_x_mat(t_v, p_d,0) #Creates a design matrix for an OLS optimization problem. 
        betax = get_coef_vec(des_mat, x_lat) #Gets coefficient vector for coordinate x approximation. 
        betay = get_coef_vec(des_mat, y_lon) #Gets coefficient vector for coordinate y approximation. 

        appr_x = implemented_function('appx_x', lambda t: get_apprx(betax, t, p_d))
        appr_x = sp.lambdify(t, appr_x(t)) #This and the previous line defines a general function for the x approximation. 

        appr_y = implemented_function('appr_y', lambda t: get_apprx(betay, t, p_d))
        appr_y = sp.lambdify(t, appr_y(t)) #This and the previous line defines a general function for the y approximation. 

        x_f = appr_x(t_grid) #Evaluates fitted funtion on the desired grid for x. 
        y_f = appr_y(t_grid) #Evaluates fitted function on the desired grid for y. 
        x_vals.append(x_f) #Adds fitted values on grid to x_vals. 
        y_vals.append(y_f) #Adds fitted values on grid to y_vals. 

        pred_x = appr_x(t_v) #Evaluates approximation polinomial on original values t_v for x. 
        pred_y = appr_y(t_v) #Evaluates approximation polinomial on original values t_v for y. 

        err_x_c = pred_x - x_lat #Computes errors for x.
        err_y_c = pred_y - y_lon #Computes errors for y.

        err_x.append(sum(err_x_c**2)) #Adds sum of squared errors for x.
        err_y.append(sum(err_y_c**2)) #Adds sum of squared errors for y. 

        axs[1,0].plot(t_grid, x_f)
        axs[1,1].plot(t_grid, y_f)
        axs[1,2].plot(x_f, y_f)
        print("Done with flight ", n)
    except:
        print(f"Error with flight {n} !!!")


dfs = write_fin_df(x_vals, y_vals, f_ids, t_grid) #Obtains data frames with approximations evaluated on equally spaced grid. 
dfs[0].write_csv(x_lat_file) #Stores latitude x data on a csv file. 
dfs[1].write_csv(y_lon_file) #Stores longitude y data on a csv file. 

print(dfs[0])
print(dfs[1])



plt.show()
print("Errors for x")
plt.hist(err_x)
plt.show()

print("Errors for y")
plt.hist(err_y)
plt.show()




max_x_idx = err_x.index(max(err_x) )
err_x.pop(max_x_idx)

max_y_idy = err_y.index( max(err_y) )
err_y.pop(max_y_idy)

plt.show()
print("Errors for x")
plt.hist(err_x)
plt.title('Errores al cuadrado, Latitud X')
plt.show()

print("Errors for y")
plt.hist(err_y)

plt.title('Errores al cuadrado, Longitud Y')
plt.show()


print("Were done!")