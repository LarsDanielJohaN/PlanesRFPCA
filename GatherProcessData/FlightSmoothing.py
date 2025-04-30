"""
Written by: Lars Daniel Johansson Niño
Created date: 09/08/2024
Purpose: Create Bsplines for functional data and returns their images under an equally spaced on [0,1]
Notes: 
"""
print("Hello GatherData3d2!")

import polars as pl
import matplotlib.pyplot as plt 
import numpy as np
import os

file_p = (os.path.realpath(__file__)).replace('FlightSmoothing.py', '') #Gets current file path.
os.chdir(file_p) #Changes working directory to current file´s path. 
cwd = os.getcwd() #Reads the new working directory. 
open_file = 'less_p_data_flights_egll_esgg1d2.csv'
save_file_name = 'smoothed_less_p_data_flights_egll_esgg1d2.csv'

np.random.seed(201763)

"""
m: value such that m+1 is the degree of B^m(x)
i: element in the basis
T: vector of knots
x: value to evaluate. 
"""
def eval_B_Spline(T, i, m, x):
    if m == -1:
        return np.float64( ( T[i-1] <= x ) and (x < T[i]))
    else:
        z0 = (x - T[i-1])/(T[i+m]- T[i-1])
        z1 = (T[i+m+1] -x)/(T[i+m+1]- T[i])
        return z0*eval_B_Spline(T, i, m-1, x) + z1*eval_B_Spline(T, i+1, m-1, x)
    
"""
get_des mat: given a collection of evaluation points x_e, obtains the design matrix of a B-Spline basis of order m+1, k basis functions
and values a,b for the smallest and biggest knots respectivelly. 

x_e: points where to evaluate functions.
k: number of elements in basis.
m: number such that m+1 is the degree of the resulting polinomial.
a,b: [a,b] a is the place where the first knot goes and b is where the second one goes. 
"""

def get_des_mat(x_e, k, m, a,b):
    T = np.linspace(a,b, m+k+2)
    B1 = np.vectorize(lambda x: eval_B_Spline(T, 1, m, x))
    X = B1(x_e)

    for i in range(2, k+1):
        Bi = np.vectorize(lambda x: eval_B_Spline(T, i, m, x))
        X = np.vstack( (X, Bi(x_e)))
    return X.T


"""
get_pen_mat creates penalty matrix on second squared diferences on the coeficeints. 

"""
def get_pen_mat(k):
    P = np.zeros((k-1,k))
    for i in range(k-1):
        P[i, i] = -1
        P[i,i+1] = 1
    return P.T@P



def get_poppers(x, min_max_p, oth_p, pop_jump):

    no_jumps =  int(np.floor(len(x)/pop_jump)) #Gets number of jumps. 
    popper = np.ones(len(x)) #Gets vector of ones to define which values to keep. 
    k_low = 0 
    k_upp = pop_jump 

    for j in range(no_jumps):
        curr = x[k_low: k_upp] #Gets from the k_low, to the k_upp elements. 
        curr_sort_arg = np.argsort(curr).astype(int) #Gets the indexes of sorted elements in curr. 


        sd_x = np.var(curr) #Computes variance of elements in curr. 
        sd_min = np.var(curr[curr_sort_arg[1:]]) #Computes the variance of elements in curr, without the smallest. 
        sd_max = np.var(curr[curr_sort_arg[:-1]]) #Computes the variance of elements in curr, without the largets. 
        sd_min_max = np.var(curr[curr_sort_arg[1:-1]]) #Computes the variance of elements in curr, without smallest and largest. 

        if sd_min_max == 0:
            sd_min_max = 1
        if sd_min == 0:
            sd_min = 1
        if sd_max == 0:
            sd_max = 1


        #print({'min':sd_x/sd_min, 'max':sd_x/sd_max, 'min_max ':sd_x/sd_min_max  })



        if sd_x/sd_min_max >= min_max_p: #If quotient of variances is bigger than threshold. See if there is something to remove. 

            if sd_x/sd_min >= oth_p: #If the variance quotient with the minimum exceds threshold, remove the lowest value. 
                popper[k_low:k_upp][curr_sort_arg[0]] = 0
            if sd_x/sd_max >= oth_p: #If the variance quotient with the maximim exceds thereshold, remove the largest value. 
                popper[k_low:k_upp][curr_sort_arg[-1]] = 0
        k_low+= pop_jump
        k_upp+= pop_jump
    
    if len(x)/pop_jump > 1 and no_jumps > 0:
        curr = x[len(x)- pop_jump: len(x)]
        curr_sort_arg = np.argsort(curr) #Gets the indexes of sorted elements in curr. 

        sd_x = np.std(curr) #Computes variance of elements in curr. 
        sd_min = np.std(curr[curr_sort_arg[1:]]) #Computes the variance of elements in curr, without the smallest. 
        sd_max = np.std(curr[curr_sort_arg[:-1]]) #Computes the variance of elements in curr, without the largets. 
        sd_min_max = np.std(curr[curr_sort_arg[1:-1]]) #Computes the variance of elements in curr, without smallest and largest. 

        if sd_x/sd_min_max >= min_max_p: #If quotient of variances is bigger than threshold. See if there is something to remove. 

            if sd_x/sd_min >= oth_p: #If the variance quotient with the minimum exceds threshold, remove the lowest value. 
                popper[len(x)- pop_jump:len(x)][curr_sort_arg[0]] = 0
            if sd_x/sd_max >= oth_p: #If the variance quotient with the maximim exceds thereshold, remove the largest value. 
                popper[len(x) - pop_jump:len(x)][curr_sort_arg[-1]] = 0

    return popper.astype(bool)
  

def get_new_obs(cf, subs,  k, m, a, b, lam, poppers ):
    
    
    cf_t = cf['t'].to_numpy()
    cf_x = cf[f'x_{subs}'].to_numpy() #Ojo aqui el cambio. 
    cf_y = cf[f'y_{subs}'].to_numpy()
    cf_z = cf[f'z_{subs}'].to_numpy()

    cf_t = cf_t[poppers]
    cf_x = cf_x[poppers]
    cf_y = cf_y[poppers]
    cf_z = cf_z[poppers]

    


    Xdm = get_des_mat(cf_t, k,m, a, b)
    D = get_pen_mat(k)


    Bx = np.linalg.pinv(Xdm.T@Xdm + lam*D)@Xdm.T@cf_x
    By = np.linalg.pinv(Xdm.T@Xdm + lam*D)@Xdm.T@cf_y
    Bz = np.linalg.pinv(Xdm.T@Xdm + lam*D)@Xdm.T@cf_z

    x_new = X_des_com@Bx
    y_new = X_des_com@By
    z_new = X_des_com@Bz

    a_x = cf_x - Xdm@Bx 
    a_y = cf_y - Xdm@By
    a_z = cf_z - Xdm@Bz 

    err_sqrd_x =  a_x@a_x.T
    err_sqrd_y = a_y@a_y.T
    err_sqrd_z = a_z@a_z.T

    return {'x_new':x_new, 'y_new':y_new, 'z_new':z_new, 'ssr_x':err_sqrd_x, 'ssr_y':err_sqrd_y, 'ssr_z':err_sqrd_z}


err_sqrd_x_org = np.array([])
err_sqrd_y_org = np.array([])
err_sqrd_z_org = np.array([])


err_sqrd_xu = np.array([])
err_sqrd_yu = np.array([])
err_sqrd_zu = np.array([])

err_sqrd_x_u0 = np.array([])
err_sqrd_y_u0 = np.array([])
err_sqrd_z_u0 = np.array([])

err_sqrd_x_r0 = np.array([])
err_sqrd_y_r0 = np.array([])
err_sqrd_z_r0 = np.array([])


err_sqrd_x_s0 = np.array([])
err_sqrd_y_s0 = np.array([])
err_sqrd_z_s0 = np.array([])


curr_data = pl.read_csv(open_file, has_header= True)

flights = curr_data['n'].unique().to_list()
T = np.linspace(0,1,  400)
k = 25
m = 2


print(curr_data.columns)
curr_data = curr_data.rename( {'x_lat':'x_org', 'y_lon':'y_org', 'z_alt':'z_org'}) #Renames latitude and longitude colums to x_lat and y_lon respectivelly. 

cols1 = [ 'x_org', 'y_org', 'z_org',  'x_u', 'y_u', 'z_u', 'x_u0', 'y_u0', 'z_u0', 'x_r0', 'y_r0', 'z_r0',  'x_s0', 'y_s0', 'z_s0',  't' ]
cols_series = [pl.Series(i, dtype = pl.Float64) for i in cols1] + [pl.Series('id', dtype = pl.String), pl.Series('cs', dtype = pl.String),pl.Series('n', dtype = pl.Int64)]
fin_data = pl.DataFrame( cols_series )
#fin_data = pl.DataFrame( [pl.Series('x', dtype = pl.Float64), pl.Series('y', dtype = pl.Float64),pl.Series('z', dtype = pl.Float64), pl.Series('t', dtype = pl.Float64), pl.Series('id', dtype = pl.String), pl.Series('cs', dtype = pl.String),pl.Series('n', dtype = pl.Int64)]  )

X_des_com = get_des_mat(T, k, m, -0.5, 1.5)


for f in flights:

    cf = curr_data.filter(pl.col('n') == f) #Selects those values with number n.
    cf_cs = cf['call_s'][0]
    cf_id = cf['id'][0]

    popper = get_poppers( cf['z_org'].to_numpy(), 2.0, 1.5, 20)
    subs_org = get_new_obs(cf, 'org', k, m, -0.5, 1.5, 1/50, popper )

    err_sqrd_x_org = np.append(   err_sqrd_x_org,  (subs_org['ssr_x'])  )
    err_sqrd_y_org = np.append(   err_sqrd_y_org,  (subs_org['ssr_y'])  )
    err_sqrd_z_org = np.append(   err_sqrd_z_org,  (subs_org['ssr_z'])  )



    subs_unit = get_new_obs(cf, 'unit', k, m, -0.5, 1.5, 1/50 , popper)

    err_sqrd_xu = np.append(   err_sqrd_xu,  (subs_unit['ssr_x'])  )
    err_sqrd_yu = np.append(   err_sqrd_yu,  (subs_unit['ssr_y'])  )
    err_sqrd_zu = np.append(   err_sqrd_zu,  (subs_unit['ssr_z'])  )


    subs_u0 = get_new_obs(cf, 'unit_0', k, m, -0.5, 1.5, 1/50, popper )
    err_sqrd_x_u0 = np.append(   err_sqrd_x_u0,  (subs_u0['ssr_x'])  )
    err_sqrd_y_u0 = np.append(   err_sqrd_y_u0,  (subs_u0['ssr_y'])  )
    err_sqrd_z_u0 = np.append(   err_sqrd_z_u0,  (subs_u0['ssr_z'])  )

    subs_r0 = get_new_obs(cf, 'r0', k, m, -0.5, 1.5, 1/50 , popper)
    err_sqrd_x_r0 = np.append(   err_sqrd_x_r0,  (subs_r0['ssr_x'])  )
    err_sqrd_y_r0 = np.append(   err_sqrd_y_r0,  (subs_r0['ssr_y'])  )
    err_sqrd_z_r0 = np.append(   err_sqrd_z_r0,  (subs_r0['ssr_z'])  )

    subs_s0 = get_new_obs(cf, 'sphr', k, m, -0.5, 1.5, 1/50 , popper)
    err_sqrd_x_s0 = np.append(   err_sqrd_x_s0,  (subs_r0['ssr_x'])  )
    err_sqrd_y_s0 = np.append(   err_sqrd_y_s0,  (subs_r0['ssr_y'])  )
    err_sqrd_z_s0 = np.append(   err_sqrd_z_s0,  (subs_r0['ssr_z'])  )





    cols1 = [ 'x_org', 'y_org', 'z_org',  'x_u', 'y_u', 'z_u', 'x_u0', 'y_u0', 'z_u0', 'x_r0', 'y_r0', 'z_r0',  'x_s0', 'y_s0', 'z_s0' 't' ]


    data_new_cf = pl.DataFrame( {'x_org':subs_org['x_new'], 'y_org':subs_org['y_new'], 'z_org':subs_org['z_new'],
                                 'x_u':subs_unit['x_new'], 'y_u':subs_unit['y_new'], 'z_u':subs_unit['z_new'],
                                 'x_u0':subs_u0['x_new'], 'y_u0':subs_u0['y_new'], 'z_u0':subs_u0['z_new'],
                                 'x_r0':subs_r0['x_new'], 'y_r0':subs_r0['y_new'], 'z_r0':subs_r0['z_new'],
                                 'x_s0':subs_s0['x_new'], 'y_s0':subs_s0['y_new'], 'z_s0':subs_s0['z_new'],
                                  't':T} )
    

    data_new_cf = data_new_cf.with_columns(pl.lit(cf_id).alias("id"), pl.lit(cf_cs).alias("cs"), pl.lit(f).alias("n"))
    data_new_cf = data_new_cf.with_columns(pl.col("n").cast(pl.Int64))
    fin_data.extend(data_new_cf)



    print(f"Done with flight {f}")















fix_err, ax_err = plt.subplots(nrows = 4, ncols = 3)
ax_err[0,0].hist(err_sqrd_x_org)
ax_err[0,0].set_title('x_org')

ax_err[0,1].hist(err_sqrd_y_org)
ax_err[0,1].set_title('y_org')

ax_err[0,2].hist(err_sqrd_z_org)
ax_err[0,2].set_title('z_org')


ax_err[1,0].hist(err_sqrd_x_u0)
ax_err[1,0].set_title('x_u0')

ax_err[1,1].hist(err_sqrd_y_u0)
ax_err[1,1].set_title('y_u0')

ax_err[1,2].hist(err_sqrd_z_u0)
ax_err[1,2].set_title('z_u0')



ax_err[2,0].hist(err_sqrd_x_r0)
ax_err[2,0].set_title('x_r0')

ax_err[2,1].hist(err_sqrd_y_r0)
ax_err[2,1].set_title('y_r0')


ax_err[3,2].hist(err_sqrd_z_r0)
ax_err[3,2].set_title('z_r0')

ax_err[3,0].hist(err_sqrd_xu)
ax_err[3,0].set_title('x_u')

ax_err[3,1].hist(err_sqrd_yu)
ax_err[3,1].set_title('y_u')

ax_err[3,2].hist(err_sqrd_zu)
ax_err[3,2].set_title('z_u0')

plt.show()


fix_err, ax_err = plt.subplots(nrows = 1, ncols = 3)
ax_err[0].hist(np.delete(err_sqrd_x_org, np.argmax(err_sqrd_x_org) ))
ax_err[0].set_title('x')
ax_err[1].hist(np.delete(err_sqrd_y_org, np.argmax(err_sqrd_y_org) ))
ax_err[1].set_title('y')

ax_err[2].hist(np.delete(err_sqrd_z_org, np.argmax(err_sqrd_z_org) ))
ax_err[2].set_title('z')

print(np.mean(err_sqrd_x_r0))

plt.show()

print(save_file_name)
fin_data.write_csv(save_file_name)






"""

D = np.linspace(0,20, 4000 )
f = lambda x : np.sin( np.exp( -np.cos(x-7)   )) -np.arctan(x-7) + 3
f = np.vectorize(f)

f_e = f(D)

j = 13

D_n = D[:: j]
f_e_n = f_e[:: j] + np.random.normal(loc = 0, scale = 1, size = D_n.shape)

k = 30
m = 2


X = get_des_mat(D_n, k, m, -10, 30)
print(len(D_n))

r,c = X.shape

P = np.diag(np.zeros(c))


for i in range(1, c ):
    P[i-1, i]= 1
    P[i-1,i-1] = -1



P = P.T@P
lam = 0.2

B = np.linalg.pinv( X.T@X + lam*P  )@X.T@f_e_n


plt.plot(D, f_e, color = 'black')
plt.scatter(D_n, f_e_n, color = 'gray')
plt.plot(D_n, X@B, color = 'red')
plt.show()



Some preeliminary testing. 
m = 2 #Deg of polinomial m+1
k = 2 #Two elements in the basis of splines. 
T = np.linspace(0, 10, m+k+2)


f, ax = plt.subplots(1,1)
D = np.linspace(0,10, 100)


for i in range(1,k+1):
    Bi = np.vectorize(lambda x: eval_B_Spline(T, i, m, x) )
    ax.plot(D, Bi(D))

plt.show()

f, ax = plt.subplots(1,1)
X = get_des_mat(D, k, m, 0, 10)
r,c = X.shape

print(X.shape)
for n in range(c):
    print(n)
    Bi = np.vectorize(lambda x: eval_B_Spline(T, n+1, m, x) )

    ax.plot(D, X[:, n]-Bi(D))
plt.show()



"""