#Created by: Lars Daniel Johansson Niño
#Last edited date: 31/5/23
#Purpose: Generate test flight
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString
from geopy import distance
import time
import math
import numpy as np
from GenerateHorDist import *
from pathos.multiprocessing import ProcessPool
from multiprocessing import Pool, freeze_support





def __main__():
    print("Hello test flight!")
    path_flight_data = r"/Users/larsdanieljohanssonnino/Library/CloudStorage/OneDrive-INSTITUTOTECNOLOGICOAUTONOMODEMEXICO/RUIDO_COMIPENS_Archivos/Ruido_Comipens_Tests/Codes/Test_code_Python/Trial31.csv"
    path_schools_cdmx = r"/Users/larsdanieljohanssonnino/Library/CloudStorage/OneDrive-INSTITUTOTECNOLOGICOAUTONOMODEMEXICO/RUIDO_COMIPENS_Archivos/Ruido_Comipens_Tests/CreatedData/G_SEP_SIGEL/2_CDMX_MIDDLE_SCHOOLS.csv"
    
    flight_data = pd.read_csv(path_flight_data)
    school_data = pd.read_csv(path_schools_cdmx)

    pool = ProcessPool()

   


    pd.set_option('display.float_format', '{:.200f}'.format)


    school_data['TRAD_POINT_WGS84'] = [(school_data['WGS84_LAT'][p], school_data['WGS84_LON'][p]) for p in school_data.index]

    center_point_mmmx = (19.4363,-99.0721)
    flights = flight_data["callsign"].unique()
    flights_to_test = []



    #Maybe, I could replace this part with some filter function
    #This part of the code selects flights which are at most 20 nautical miles from MMMX, the flight_data data frame already contains flights at an altitude of at most 15000 feet. 
    #That should be enought to eliminating the possibility of having en-route flights. 
    for i in flights:
        test_flight = flight_data[flight_data["callsign"] == i]
        aux_point = (test_flight.iloc[0]['latitude'],test_flight.iloc[0]['longitude'])

        if math.isnan(test_flight.iloc[0]['latitude']) or math.isnan(test_flight.iloc[0]['longitude']):
            print("Check for flight: ",i)
        else:
            dist = distance.distance(aux_point,center_point_mmmx).meters
            if dist/1852 <=20:
                flights_to_test.append(i)


    #Vectorized values for school coordinates, altitudes, and their respective id´s
    school_id_vect = school_data['CVE_CENTR_TRB'].to_numpy()
    school_cord_vect = school_data['TRAD_POINT_WGS84'].to_numpy()
    school_alt_vect = school_data['ALT_FT'].to_numpy()
    school_zip = zip(school_cord_vect, school_alt_vect, school_id_vect)
    #print(list(school_zip))
    hor_dist = GenerateHorDist(school_zip)
    data = []
    start_time_a = time.time()

    for f in flights_to_test:

        print("Going for flight: ",f)
        flight_f_data = flight_data[flight_data["callsign"] == f]
        start_time_b = time.time() #Starts timer to keep track of execution time
        flight_points = [(flight_f_data['latitude'][t], flight_f_data['longitude'][t])for t in flight_f_data.index] # Generates points captured in a particular time interval

        aux = list(hor_dist.calculate_hor_distances(flight_points))
        print(aux)
        break 


        
       
    
        end_time_b = time.time()
        print("Flight",f,"took ",(end_time_b-start_time_b)," seconds to process.")
      
      
    end_time_a  = time.time()
    print("Total process took",(end_time_a-start_time_a),"seconds")
    print("Done with execution")




if __name__ == '__main__':
    freeze_support()
    __main__()



"""
        #flight_alt_vect = [len(school_data.index)*[flight_f_data['altitude'][t]]] #Creates a vector of the size of our schools with the current altitude of our flight
            #alt_dif_vect_i = np.subtract(flight_alt_vect,school_alt_vect) #Creates the m schools diference observatiosn at that particular point
            

    #processed_data_day["FLIGHT_ALT"].append(flight_alt_vect) #Adds the current flight altitude, to be formated later to SQL
           # processed_data_day['ALT_DIST'].append(alt_dif_vect_i) #Adds the current altitude distance between flight and schools


        processed_data_day["callsign"].append([[f] * (len(school_data.index)*len(flight_f_data.index))])  #Adds the current flight callsign, to be formated later to SQL
        processed_data_day['icao24'].append( [[flight_f_data.iloc[0]['icao24']]*(len(school_data.index)*len(flight_f_data.index))])   #Adds the current flight icao24 code, to be formated later to SQL
        processed_data_day["CVE_CENTR_TRB"].append(school_id_vect*len(flight_f_data.index)) # Adds the different school´s id´s 
 

"""