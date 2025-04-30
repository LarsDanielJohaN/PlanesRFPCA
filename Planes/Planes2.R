#Writen by: Lars Daniel Johansson Niño
#Created date: 08/08/2024
#Purpose: Perform FPCA with flight data. 


library(refund)
library(dplyr)
library(tidyverse)
library(RFPCA)

fd <- as.data.frame(read.csv('smoothed_less_p_data_flights_egll_esgg1d2.csv')  ) #Reads data on processed curves, latitude. 

obs_xy <- fd %>% select(n, x_org, y_org)
obs_t <- fd%>%select(n, t)

n_un <- max(unique(obs_xy$n))
obs_xy_list <- list()
obs_t_list <- list()

for(i in 1:n_un){
  curr_n <- t(obs_xy%>%filter(n == i)%>%select(x_org, y_org))
  curr_t <- t(obs_t%>%filter(n == i)%>%select(t))
  rownames(curr_n) <- NULL
  rownames(curr_t) <- NULL
  
  obs_xy_list[[i]] <- curr_n 
  obs_t_list[[i]] <- c(curr_t)
  
}



l <- dim(curr_t)[2]

bw <- 00.1
kern <- 'gauss'
K <- 12
nRegG = 80

mfd <- structure(1, class='Euclidean')
resEu <- RFPCA(obs_xy_list, obs_t_list, list(userBwMu=bw, userBwCov=bw * 2, npoly = 3,  nRegGrid = nRegG, kernel=kern, maxK=K, mfd=mfd)) #Performs RFPCA on sparsified observations. 



mu_est_eu <- t(resEu$muWork)
phi_eu <- resEu$phi

eu_obs_df <- data.frame(x = mu_est_eu[,1], y = mu_est_eu[,2], lbl = rep('mu_est', length(mu_est_eu[,1]))    )


print("Principal components")

for(i in 1:resEu$K){
  print("------------------------------------------------------------------------")
  
  c_phi <- phi_eu[,,i]
  curr_df <- data.frame(x = c_phi[,1], y = c_phi[,2],    lbl =  rep(paste('phi',as.character(i)), length(c_phi[,1]))        )

  
  eu_obs_df <- rbind(eu_obs_df, curr_df)
}


var_lam_eu <- resEu$lam
pve_ish_eu <- var_lam_eu/sum(var_lam_eu)
pve2 <- cumsum(pve_ish_eu)

pve2_k_d <- formatC(pve2, digits = 4, format = "fg", flag = "#")

write.csv(eu_obs_df, 'fpca_org_flights_egll_esgg1d2.csv')


matplot( resEu$muObs[1,], resEu$muObs[2,], type='l', lty=2)

resEu$muReg