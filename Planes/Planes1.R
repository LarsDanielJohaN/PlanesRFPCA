#Writen by: Lars Daniel Johansson Niño
#Created date: 08/08/2024
#Purpose: Perform FPCA with flight data. 


library(refund)
library(tidyverse)
library(RFPCA)
library(abind)
library(pracma)
library(Rmpfr)
library(ggplot2)

options(digits = 22)  # or any number up to 22


fd <- as.data.frame(read.csv('smoothed_less_p_data_flights_egll_esgg1d2.csv')  ) #Reads data on processed curves, latitude. 
a <- pi/180
fd <- fd%>%mutate(x_s0 = sin(a*x_org)*cos(a*y_org), y_s0 = sin(a*x_org)*sin(a*y_org), z_s0 = cos(a*x_org) )
obs_xyz <- fd%>%select(n, x_s0, y_s0, z_s0)

obs_t <- fd%>%select(n, t)

no_t <- length(unique(obs_t$t))
n_un <- max(unique(obs_xyz$n))
obs_xyz_list <- list()
obs_t_list <- list()

for(i in 1:n_un){
  curr_n <- t(obs_xyz%>%filter(n == i)%>%select(x_s0, y_s0, z_s0))
  curr_t <- t(obs_t%>%filter(n == i)%>%select(t))
  rownames(curr_n) <- NULL
  rownames(curr_t) <- NULL
  
  obs_xyz_list[[i]] <- curr_n 
  obs_t_list[[i]] <- c(curr_t)
  
}

bw <- 00.1
kern <- 'gauss'
K <- 12
nRegG = 100
mfdSp <- structure(1, class='Sphere')
resSp <- RFPCA(obs_xyz_list, obs_t_list, list(userBwMu=bw, userBwCov=bw * 2, npoly = 3,  nRegGrid = nRegG, kernel=kern, error = FALSE, maxK=K, mfd=mfdSp)) #Performs RFPCA on sparsified observations. 



mu_est <- t(resSp$muWork)
mu_org_grid <- t(resSp$muObs) #Gets estimated mean function on matrix of dimension (# points in grid) x 3

norm_mu_sphr <- sqrt(mu_org_grid[,1]^2 + mu_org_grid[,2]^2 + mu_org_grid[,3]^2)
is_on_sphere_mu_sphr <- abs(norm_mu_sphr - 1) < 0.0001



phi <- resSp$phi # nWorkGrid x D x K

to_sphere <- function(mu, phi, l){
  phi_fin <- phi
  
  for(i in 1:l){
    v <- phi_fin[i, ]
    curr_mu <- mu[i,]
    v_n <- sqrt(sum(v^2) )
    phi_fin[i,] <- cos(v_n)*curr_mu + sin(v_n)*v*(1/v_n)
  }
  phi_fin
}



sph_obs_df <- data.frame(x = mu_est[,1], y = mu_est[,2], z = mu_est[,3], lbl = rep('mu_est', length(mu_est[,1])    )    ) #Stores mean function in appropiate dataframe. 


for (i in 1:resSp$K){
  print("------------------------------------------------------------------------------------------------------")
  c_phi <- phi[,,i]
  s_phi <- to_sphere(mu_est, (1 )*c_phi, nRegG) #3*sqrt(resSp$lam[i])  
  
  aux <- s_phi[,1]^2 + s_phi[,2]^2 + s_phi[,3]^2
  print(mean(aux))
  print(var(aux))
  
  curr_df <- data.frame(x = s_phi[,1], y = s_phi[,2], z = s_phi[,3],lbl = rep(paste('phi',as.character(i)), length(s_phi[,1])))
  curr_df_e <-data.frame(x = c_phi[,1], y = c_phi[,2], z = c_phi[,3],lbl = rep(paste('phi_eu',as.character(i)), length(s_phi[,1]))      )
  sph_obs_df <- rbind(sph_obs_df, curr_df)
  sph_obs_df <- rbind(sph_obs_df, curr_df_e)
}


write.csv(sph_obs_df, 'rfpca_spherical_flights_egll_esgg1d2.csv')


xi <- resSp$xi #Matrix of dimensions n_un x k, i.e. its element [i, ] corresponds to the estimated scores for the i-th observed function. 
phi_org_grid <- resSp$phiObsTrunc #Gets tensor of dimensions (# points in grid) x 3 x k, i.e. its element [,,i] corresponds to the i-th estimated eigenfunction. 
mu_org_grid <- t(resSp$muObs) #Gets estimated mean function on matrix of dimension (# points in grid) x 3


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


phi_acc <- lapply(rep(1, n_un), function(x){mu_org_grid}) #Creates a vector with the mean function repeated n_un times i.e. phi_acc_0[[i]] = mu_org_grid for i = 1,2,...., n_un
phi_acc <- abind(phi_acc, along = 3) #Converts phi_acc to a tensor of dimensions (# points in grid) x 3 x n_un 
phi_acc <- array(phi_acc, dim = c(as.vector(dim(phi_acc)),1)) #Converts phi_acc to a tensor of dimensions (# points in grid) x 3 x n_un x 1. i.e. to later store everything in the same tensor. 



for(i in 1:K){
  phi_acc_curr <- lapply(xi[,i], function(x){x*phi_org_grid[,,i]}) #Multiples individual observation scores by i-th estimated eigenfunction. 
  phi_acc_curr <- abind(phi_acc_curr, along = 3) #Converts phi_acc_curr to a tensor of dimensions (# points in grid) x 3 x n_un
  phi_acc_curr <- array(phi_acc_curr, dim = c(as.vector(dim(phi_acc_curr)),1))#Converts phi_acc_curr to a tensor of dimensions (# points in grid) x 3 x n_un x 1. i.e. to later store everything in the same tensor. 
  phi_acc <- abind(phi_acc, phi_acc_curr, along = 4) #Adds phi_acc_curr to phi_acc as a new slice. 
} #Obs: in the end, phi_acc will be a tensor of dimension (#points in grid) x 3 x n_un x K. 


geodesic_dist_sphere <- function(p, q, l){ #p, and q are two matrices of the same dimensions

  warn <- 0
  prob_x <- 0
  inn_prods <- rep(0,l)
  toRet <- rep(0, l) #Creates vector of zeros to store output. 
  for( i in 1:l){
    inn_prod <- sum(p[i,]*q[i,]) #Takes inner product between p[i,] and q[i,]
    inn_prods[i] <- inn_prod
    toRet[i] <- acos(inn_prod) #Takes spherical distance between ith rows p[i,] and q[i,]
    if(is.nan(toRet[i]) ){
      
      if(sqrt(sum(p[i,]^2))>1  | sqrt(sum(p[i,]^2))<1 ){
        prob_x <- prob_x + 1
      }
      warn <- warn + 1
      if(inn_prod > 1)
        toRet[i] <- acos(1)
      else 
        toRet[i] <- acos(-1)
    }
      
  }
  
  #May I be forgiven for this way of programming. Warning, warn_track and n_curr are modified within the geodesic_dist_sphere function. 
  warn_track_curr <- data.frame(n = c(n_curr), warn = c(warn), prop = c(warn/l), k_n = c(k_curr), prob_x = c(prob_x), diff_warn_x = c(abs(warn-prob_x)) , max_inn_prod = c(max(inn_prods)), min_inn_prod = c(min(inn_prods)))
  warn_track <<- rbind(warn_track, warn_track_curr)
  n_curr <<- n_curr + 1
  
  return (toRet)

 } #Asumes that both 


warn_track <- data.frame(n = c(0), warn = c(0), prop = c(0), k_n = c(0), prob_x = c(0), diff_warn_x = c(0), max_inn_prod = c(NaN), min_inn_prod = c(NaN))
k_curr <- 0
n_curr <- 1

obs_X <- abind(obs_xyz_list, along = 3)
norm_X <- matrix(0, nrow = n_un, ncol = no_t)

for(i in 1:n_un){
  curr_X <- obs_X[,,i]
  norm_X[i, ] <- sqrt( curr_X[1,]^2 + curr_X[2,]^2  +curr_X[3,]^2 )
}


hist(norm_X)
max(norm_X)
min(norm_X)


geod_dist <- lapply(1:n_un, function(x){  geodesic_dist_sphere(t(obs_X[,,x]), mu_org_grid, no_t )         })
geod_dist <- abind(geod_dist, along = 2)
geod_dist <- array(geod_dist, dim = c(as.vector(dim(geod_dist)), 1))
  
aux_ten <-array(0, c(no_t, 3, n_un))#Creates auxiliary tensor to accumulate xi*phi(t) values up to the k-th eigenfunction. 

for(k in 1:K){
  
  k_curr <- k
  n_curr <- 1
  
  curr_acc_phi <- phi_acc[,,,k+1] #Obtains values xi_k phi_k(t) for the original grid. 
  aux_ten <- aux_ten + curr_acc_phi #Obtains the previous + xi_k phi_k(t)
  curr_acc_k_sphr <- lapply(1:n_un, function(x){  to_sphere(mu_org_grid, aux_ten[,,x], no_t)})#Passes sum(xi_k phi_k(t)) to the sphere. 
  curr_acc_k_sphr <- abind(curr_acc_k_sphr, along = 3) #Accomodates the former as a tensor of dimension no_t x 3 x n_un for better management.
  l <- sqrt(curr_acc_k_sphr[,1,]^2 + curr_acc_k_sphr[,2,]^2 + curr_acc_k_sphr[,3,]^2) #Computes pointwise norm to see whether points are on the unit sphere. 
  lmean <-apply(l, 2, mean) #Calculates, by observation, the mean of the norm values.
  lvar <- apply(l, 2, var) #Calculates, by observation, the variance of the norm values.
  
  curr_geod_dist <- lapply(1:n_un, function(x){geodesic_dist_sphere( t(obs_X[,,x])  , curr_acc_k_sphr[,,x] ,no_t)}) #Calculates, pointwise, the geodesic distance for exp(sum xi_k phi_k(t)) to the actual observation. 
  curr_geod_dist <- abind(curr_geod_dist, along = 2) #Accomodates the formeer as a tensor of dimension no_t x n_un for better management. 
  #phi_acc <- abind(phi_acc, phi_acc_curr, along = 4) #Adds phi_acc_curr to phi_acc as a new slice. 
  geod_dist <- abind(geod_dist, curr_geod_dist, along = 3)#Adds curr_dist_sqrd to dist_sqrd as a new slice.
  print("..........................................................................................................")
  print("mean of means")
  print(mean(lmean))
  print("Mean of variances")
  print(mean(lvar))
  print("..........................................................................................................")
}


ggplot(warn_track, aes(x =warn)) +geom_histogram(bins = 7, fill = "steelblue") +facet_wrap(~ k_n) +theme_minimal()
ggplot(warn_track, aes(x =prob_x)) +geom_histogram(bins = 7, fill = "blue") +facet_wrap(~ k_n) +theme_minimal()
ggplot(warn_track, aes(x =diff_warn_x)) +geom_histogram(bins = 7, fill = "cyan") +facet_wrap(~ k_n) +theme_minimal()
hist(warn_track$max_inn_prod)
hist(warn_track$min_inn_prod)

#--------------------------------------------------------------------------------------------------------------------------------------------------

dist_sqrd <- matrix(0, nrow = n_un, ncol = K + 1)

for(t in 1:no_t){
  dist_sqrd <- dist_sqrd + geod_dist[t, ,]^2
}
U0 <- mean(dist_sqrd[,1])
U_m0 <- colMeans(dist_sqrd[, 1:K+1])
FVE <- ((U0 - U_m0)/U0)

FVE_k_d <- formatC(FVE, digits = 4, format = "fg", flag = "#")


FVE_to_95 <- FVE >= 0.95

#-----------------------------------------------------------------------------------------------------------------------
#A little sanity check, see if final calculations coincide with what they are supposed to be. 

xi <- resSp$xi #Matrix of dimensions n_un x k, i.e. its element [i, ] corresponds to the estimated scores for the i-th observed function. 
phi_org_grid <- resSp$phiObsTrunc #Gets tensor of dimensions (# points in grid) x 3 x k, i.e. its element [,,i] corresponds to the i-th estimated eigenfunction. 
mu_org_grid <- t(resSp$muObs) #Gets estimated mean function on matrix of dimension (# points in grid) x 3


u0g <- lapply(1:n_un, function(x){  geodesic_dist_sphere(t(obs_X[,,x]), mu_org_grid, no_t )         })

u0 <- 0
for( i in 1:n_un){
  u0 <- u0 + sum( u0g[[i]]^2)
}
u0 <- u0/n_un


phi1 <- phi_org_grid[,,1]
phi2 <- phi_org_grid[,,2]


xi1 <- xi[,1]
xi2 <- xi[,2]
u1 <- c()

for( i in 1:n_un){
  
  curr_sphr <- to_sphere(mu_org_grid, xi1[i]*phi1 + xi2[i]*phi2, no_t)
  curr <- geodesic_dist_sphere(t(obs_X[,,i]),  curr_sphr, no_t  )
  u1 <- c(u1, sum(curr^2))
}

u1 <- mean(u1)

curr_sphr <- to_sphere(mu_org_grid, xi1[1]*phi1, no_t)
curr <- geodesic_dist_sphere(t(obs_X[,,1]),  curr_sphr, no_t  )
l <- geod_dist[,,2]
l <- l[,2]


ll <- dist_sqrd[1,2]
aa <- sum(curr^2)




#----------------------------------------------------------------------------------------------------------------------------


mu_est_work<- t(resSp$muWork)
phi_work <- resSp$phiObsTrunc

work_obs_df <- data.frame(x = mu_est_work[,1], y = mu_est_work[,2], z = mu_est_work[,3], lbl = rep('mu_est', length(mu_est_work[,1]))    )
aux <- mu_est_work[,1]^2 + mu_est_work[,2]^2 + mu_est_work[,3]^2 


print("Principal components")

for(i in 1:resSp$K){
  print("------------------------------------------------------------------------")
  c_phi <- phi_work[,,i]
  curr_df <- data.frame(x = c_phi[,1], y = c_phi[,2], z = c_phi[,3],    lbl =  rep(paste('phi',as.character(i)), length(c_phi[,1]))        )
  
  aux <- c_phi[,1]^2 + c_phi[,2]^2 + c_phi[,3]^2
  print(mean(aux))
  print(var(aux))
  
  work_obs_df <- rbind(work_obs_df, curr_df)
}

write.csv(work_obs_df, 'rfpca_work_flights_egll_esgg1d2.csv')



#----------------------------------------------------------------------------------------------------------------------------------


var_lam <- resSp$lam
pve_ish <- var_lam/sum(var_lam)

pve <- cumsum(pve_ish)
pve_k_d <- formatC(pve, digits = 4, format = "fg", flag = "#")

pve_to_95 <- pve >= 0.95

mfd <- structure(1, class='Euclidean')
resEu <- RFPCA(obs_xyz_list, obs_t_list, list(userBwMu=bw, userBwCov=bw * 2, npoly = 3,  nRegGrid = nRegG, kernel=kern, error = FALSE,  maxK=K, mfd=mfd)) #Performs RFPCA on sparsified observations. 

mu_est_eu <- t(resEu$muWork)
mu_est_eu_obs <- t(resEu$muObs)
norm_eu_mu <- sqrt(  mu_est_eu_obs[,1]^2 + mu_est_eu_obs[,2]^2 +mu_est_eu_obs[,3]^2)
is_on_sphere_mu_eu <- abs(norm_eu_mu - 1) < 0.0001
phi_eu <- resEu$phi

eu_obs_df <- data.frame(x = mu_est_eu[,1], y = mu_est_eu[,2], z = mu_est_eu[,3], lbl = rep('mu_est', length(mu_est[,1]))    )
aux <- mu_est_eu[,1]^2 + mu_est_eu[,2]^2 + mu_est_eu[,3]^2 

print("Eucledian mean")
print(mean(aux))
print(var(aux))

print("Principal components")

for(i in 1:resEu$K){
  print("------------------------------------------------------------------------")
  c_phi <- phi_eu[,,i]
  curr_df <- data.frame(x = c_phi[,1], y = c_phi[,2], z = c_phi[,3],    lbl =  rep(paste('phi',as.character(i)), length(c_phi[,1]))        )
  
  aux <- c_phi[,1]^2 + c_phi[,2]^2 + c_phi[,3]^2
  print(mean(aux))
  print(var(aux))
  
  eu_obs_df <- rbind(eu_obs_df, curr_df)
}


var_lam_eu <- resEu$lam
pve_ish_eu <- var_lam_eu/sum(var_lam_eu)
write.csv(eu_obs_df, 'rfpca_euclidean_flights_egll_esgg1d2.csv')


matplot( t(resEu$muObs), type='l', lty=2)
matplot(t(resSp$muObs), type='l', lty=1, add=TRUE)
