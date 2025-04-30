"""
Author: Lars Daniel Johansson Niño
Created date: 02/08/2024
Purpose: Gather data for project. 

Code notes:
-We call opensky-trino to trino.opensky.org.
-This code depends on provided username and password information from trino-opensky account to the traffic library on the document settings.conf. Document isnt public. 
-Impersonation problem from trino, change username to lowercase in the settings.conf document. 
-The traffic library must be installed with conda within a partcular conda environment, and must be excecuted with its according python interpreter. (i.e. set according interpreter in VS Code)
-In the KLAX-KJFK case, failed for feb02, feb18, feb19, 2023-02-23, 10-31 march
-In the EGLL-ESGG case, jan02, jan 18, jan19, 2023-02-23, 
""" 

from traffic.data import opensky
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import os

file_p = (os.path.realpath(__file__)).replace('GatherData1.py', '') #Gets current file path.
os.chdir(file_p) #Changes working directory to current file´s path. 


def get_day(d): #Method that returns string value of day. 
    if d<10:
        return "0" + str(d)
    else:
        return str(d)

days = np.arange(1,32, 1) 
month = "03"
year = "2023"
data_min = [] #Array to store airborne time in minutes for flights during the period. 
info_att = 5 #Sets number of allowed attempts. 
dep_a = 'EGLL' #Departing airport. 
arr_a = 'ESGG' #Arrival airport. 
save_fold = 'data2'
f_n = 0
cwd = os.getcwd()
f_p_list = open('RecordedFlights2.txt', 'a' )


for d in days:
    day = get_day(d)

    date = f"{year}-{month}-{day}" #Sets current day. 
    start = f"{date} 00:00" #Year/Month/Day
    stop = f"{date} 23:59"


    trial_att =0 #Sets number of performed attempts to 0. 

    while trial_att != info_att: #Per day, performs info_att (or less) attempts to retrieve flight information for the day. 
        try: #Perform attempt to retrieve information. Could fail due to opensky-trino server errors, connection errors, etc. 
            save_f = f_n
            f= opensky.flightlist(start =start, stop = stop, departure_airport = dep_a, arrival_airport = arr_a) #Obtains flights on the specified day and the arriving and departing airports desired. 
            callsigns = f['callsign'].unique() #Obtains unique callsigns of flights
            print(f) #Prints table with retrieved flights. (This is a Traffic object from the traffic library)
            #print(callsigns) #Prints array with unique callsigns for flight. 

            p_f_info = opensky.history(start = start, stop = stop, callsign = callsigns, departure_airport= dep_a, arrival_airport= arr_a) #Gathers required flights information (all of it)            

            for i in range(0, len(p_f_info)): #Computes the airborne flight duration in minutes for stored flights and sets flight id´s. 
                f_n += 1
                ab = p_f_info[i]#Selects airborne info for flight i
                f_id =  f"\\trinoopensky_Dep{dep_a}_Arr{arr_a}_Date{year}_{month}_{day}_fn{str(f_n)}_cs" + ab.callsign + ".csv"
                #ab = ab.assign_id(name = f_id)
                ab.to_csv(filename= (cwd + f"\\{save_fold}"+ f_id))
                f_p_list.write((cwd + f"\\{save_fold}"+ f_id + "\n"))
                tot_time = ab.stop-ab.start #Computes airborne flight time (as timedelta object)
                data_min.append(tot_time.total_seconds()/60) #Stores airborne time in minutes. 
            
            
            #file_store_n = f'\\trinoopensky_Dep{dep_a}_Arr{arr_a}_Date{year}_{month}_{day}.csv' #Creates name to store flight data for the day. 
            #p_f_info.to_csv(filename=(cwd  +file_store_n) )
            #print(f"Succesful store for file {file_store_n}")
            trial_att = info_att #In case information was retrieved succesfully, set trial_att to info_att to stop attempts. 

        except:
            trial_att += 1 #In case of a failed attempt, which should direct the code to except, add one attempt. 
            f_n = save_f
            print(f"Failed attempt number {trial_att} for day {date}") #Prints message for failed attempt. 
f_p_list.close()
plt.hist(data_min) #Shows a histogram 
plt.show()




print("---------------------------------------------------------------------------------------------------------------------------------------------")
print("Were done!")






