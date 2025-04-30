import polars as pl
import matplotlib.pyplot as plt 
import numpy as np
import os

np.random.seed(201763)
file_p = (os.path.realpath(__file__)).replace('rare_val_explr.py', '') #Gets current file path.
os.chdir(file_p) #Changes working directory to current file´s path. 
cwd = os.getcwd() #Reads the new working directory.



def sample_mix_norm_cauchy(N, mu= 0, sig = 1, p_norm = 9/10):
    norm_cauchy = np.random.binomial(n = 1, p = p_norm, size = N)
    f = lambda x: x*np.random.normal(loc = mu, scale = sig, size = 1) + (1-x)*np.random.standard_cauchy(size = 1)
    f = np.vectorize(f)

    return {'noise':f(norm_cauchy), 'prov':norm_cauchy}


def variance_quotient(x):
    x_sorted = np.sort(x)

    var_x = np.var(x_sorted)
    var_x_min = np.var(x_sorted[1:len(x_sorted)])
    var_x_max = np.var(x_sorted[0:len(x_sorted)-1])
    var_x_min_max = np.var(x_sorted[1:len(x_sorted)-1])

    return {'min':var_x/var_x_min , 'max':var_x/var_x_max, 'min_max':var_x/var_x_min_max}

def get_poppers(x, min_max_p, oth_p, pop_jump):

    no_jumps =  int(np.floor(len(x)/pop_jump)) #Gets number of jumps. 
    popper = np.ones(len(x)) #Gets vector of ones to define which values to keep. 
    k_low = 0 
    k_upp = pop_jump 

    for j in range(no_jumps):
        curr = x[k_low: k_upp] #Gets from the k_low, to the k_upp elements. 
        curr_sort_arg = np.argsort(curr) #Gets the indexes of sorted elements in curr. 

        var_x = np.var(curr) #Computes variance of elements in curr. 
        var_min = np.var(curr[curr_sort_arg[1:]]) #Computes the variance of elements in curr, without the smallest. 
        var_max = np.var(curr[curr_sort_arg[:-1]]) #Computes the variance of elements in curr, without the largets. 
        var_min_max = np.var(curr[curr_sort_arg[1:-1]]) #Computes the variance of elements in curr, without smallest and largest. 

        if var_x/var_min_max >= min_max_p: #If quotient of variances is bigger than threshold. See if there is something to remove. 

            if var_x/var_min >= oth_p: #If the variance quotient with the minimum exceds threshold, remove the lowest value. 
                popper[k_low:k_upp][curr_sort_arg[0]] = 0
            if var_x/var_max >= oth_p: #If the variance quotient with the maximim exceds thereshold, remove the largest value. 
                popper[k_low:k_upp][curr_sort_arg[-1]] = 0
        k_low+= pop_jump
        k_upp+= pop_jump
    
    if len(x)/pop_jump > no_jumps and no_jumps > 0:
        curr = x[len(x)- pop_jump: len(x)]
        var_x = np.var(curr) #Computes variance of elements in curr. 
        var_min = np.var(curr[curr_sort_arg[1:]]) #Computes the variance of elements in curr, without the smallest. 
        var_max = np.var(curr[curr_sort_arg[:-1]]) #Computes the variance of elements in curr, without the largets. 
        var_min_max = np.var(curr[curr_sort_arg[1:-1]]) #Computes the variance of elements in curr, without smallest and largest. 

        if var_x/var_min_max >= min_max_p: #If quotient of variances is bigger than threshold. See if there is something to remove. 

            if var_x/var_min >= oth_p: #If the variance quotient with the minimum exceds threshold, remove the lowest value. 
                popper[len(x)- pop_jump:pop_jump][curr_sort_arg[0]] = 0
            if var_x/var_max >= oth_p: #If the variance quotient with the maximim exceds thereshold, remove the largest value. 
                popper[len(x) - pop_jump:pop_jump][curr_sort_arg[-1]] = 0

    return popper.astype(bool)
  
N = 700
f = lambda t: np.sqrt(t)*np.sin(t/np.pi) + 7
f = np.vectorize(f)

T = np.linspace(0, 35, N)
noise = sample_mix_norm_cauchy(N = N, sig = 0.0001)
cat = noise['prov']
y_obs = f(T) + noise['noise']

popper = get_poppers(y_obs, 2.0, 1.5, 20)

fig, ax = plt.subplots(nrows = 1, ncols = 2)
ax[0].scatter(T, y_obs, c = noise['prov'])
ax[1].scatter(T[popper], y_obs[popper], c = noise['prov'][popper])
plt.show()


print(y_obs.shape)

#plt.scatter(T, y_obs)
#plt.show()

k = 20
km =  0
jump = 20

for i in range(0, N, jump):
    fig,ax = plt.subplots(nrows = 1, ncols =2)
    ax[0].hist(y_obs[km:k])
    ax[1].scatter(T[km:k], y_obs[km:k], c = cat[km:k])
    print(variance_quotient(y_obs[km:k]))

    plt.show()
    km+=jump
    k+= jump    

