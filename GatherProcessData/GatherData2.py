"""
Written by: Lars Daniel Johansson Niño
Created on: 03/08/2024
Purpose: Prepare flight data for analysis 1. Filtering flights and create a common
Notes:
-This script arranges data to implement methods from the face package in R. 
-x is taken as the latitude and y as the longitude. 
- Flights with a duration of less than 13000 seconds are taken out. 
"""
print("Hello GatherData2! ")
import matplotlib.pyplot as plt 
import numpy as np
import polars as pl
import os
from datetime import datetime

file_p = (os.path.realpath(__file__)).replace('GatherData2.py', '') #Gets current file path.
os.chdir(file_p) #Changes working directory to current file´s path. 
cwd = os.getcwd() #Reads the new working directory. 

def get_day(d): #Method that returns string value of day. 
    if d<10:
        return "0" + str(d)
    else:
        return str(d)
    

def get_aux_type(time):
    if time < 50:
        return 'red'
    elif time < 70:
        return 'yellow'
    else:
        return 'lightblue'

save_folder = 'data2'
save_file_name = 'data_flights_egll_esgg1d2.csv'
filter_by_alt = False
filter_alt = 0
filter_alt2 = 50000
cols = [pl.Series('id', [], dtype = pl.String),pl.Series('t', [], pl.Float64), pl.Series('x_lat', [],  dtype= pl.Float64), pl.Series('y_lon', [], dtype= pl.Float64), pl.Series('z_alt', [], dtype = pl.Float64), pl.Series('call_s', [], dtype = pl.String), pl.Series('n', [], dtype = pl.Int64)] #Creates columns for final data. 
final_data = pl.DataFrame(cols) #Creates data frame to store final data. 
tim = [] #Creates array to explore categories of flight time. 
print(final_data)
fig,ax = plt.subplots()
n_f = 0 #Sets counter for registered number of flights. 
n_e = 0 #Sets counter for number of exceptions (those flights arent registered).


recorded_flights_f = open('RecordedFlights2.txt', 'r') #Opens file with flight paths information. 
recorded_flights = [flight[2:len(flight)-1] for flight in recorded_flights_f] #Takes into a list.
recorded_flights_f.close() #Closes RecordeFlights1.txt file.
err_flights_f = open('ErrGD22.txt', 'a')

min_date = datetime.max
max_date = datetime.min

for f_p in recorded_flights:

    try: 
        f = pl.read_csv(f_p, has_header= True) #Reads file from f_p
        id = f_p.replace((file_p + f'{save_folder}\\trinoopensky_').replace('C:', ''), '')
        id = id.replace('.csv', '')

        f = f.with_columns( f['timestamp'].str.to_datetime('%Y-%m-%d %H:%M:%S%:z' ) ) #Casts timestamp (read as string) to datetime. 
        f = f.select(['timestamp', 'latitude', 'longitude', 'altitude', 'callsign']) #Selects necessary columns. 
        f_min_date = f['timestamp'].min() 
        f_max_date = f['timestamp'].max() 


        f_min_date = f_min_date.replace(tzinfo=None)
        f_max_date = f_max_date.replace(tzinfo= None) 

        if min_date >= f_min_date:
            min_date = f_min_date

        if max_date <= f_max_date:
            max_date = f_max_date


        if filter_by_alt:
            f = f.filter((pl.col('altitude') < filter_alt2) and (pl.col('altitude') >= filter_alt)) #Filters points to only those recorded at an altitude of at least filter alt. 

        first_t = f.row(0)[0] #Selects first recorded datetime. 
        f = f.with_columns( ( pl.col('timestamp') - first_t).alias('t')   ) #Creates column with flight duration since first point above 7000 ft. 
        f = f.with_columns(pl.col('t').dt.total_seconds().cast(pl.Float64)) #Extracts the total number of seconds in duration. 
        max_secs = f.row(len(f)-1)[5] #Selects the number of seconds in last point.

        if get_aux_type(max_secs) != 'lightblue':
            raise Exception() 
        
        tim.append(max_secs)
        f = f.with_columns( [(pl.col('t')/max_secs), pl.Series('id', [id]*len(f))  , pl.Series('n', [(n_f+1)]*len(f))]) #Normalizes all t points to the unit interval, creates column with unique id. 
        f = f.rename( {'latitude':'x_lat', 'longitude':'y_lon', 'altitude':'z_alt', 'callsign':'call_s'}) #Renames latitude and longitude colums to x_lat and y_lon respectivelly. 
        f = f.select(['id', 't', 'x_lat', 'y_lon', 'z_alt','call_s' ,'n']) #Selects desired columns. 
        ax.plot(f['x_lat'].to_list(), f['y_lon'].to_list()) #Plots latitude and longitude. 
        f =  f.drop_nulls()
        final_data.extend(f)
        n_f+=1

        #plt.hist(f['z_alt'].to_list())
        #plt.show()
    
    except:
        err_flights_f.write( f_p + '\n')
        n_e += 1


print("min date,   ", min_date)
print("max_date,  ", max_date)

err_flights_f.close()

eq_rad_a = 6378137
sem_min_ax_c = 6356752.314245 


#final_data = final_data.with_columns( pl.col('z_alt')) #Changes altitude from feet to meters. 





var_eps = np.finfo(float).eps + 0.000000004

#Method one of passing to spheroid, take points and make them of norm one. 
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

final_data = final_data.with_columns( (pl.col('x_lat')/1).alias('x_lat_n') , (pl.col('y_lon')/1).alias('y_lon_n')  , (pl.col('z_alt')/100).alias('z_alt_n') )

final_data = final_data.with_columns( (pl.col('x_lat')**2).alias('x_lat_sqrd_0') , (pl.col('y_lon')**2).alias('y_lon_sqrd_0') , (pl.col('z_alt_n')**2).alias('z_alt_n_sqrd_0')    ) #Gets component wise squares. 

final_data = final_data.with_columns(  (np.sqrt( pl.col('x_lat_sqrd_0') + pl.col('y_lon_sqrd_0') + pl.col('z_alt_n_sqrd_0')  )   ).alias('norm_0')) #Gets the norm of single point. 
final_data = final_data.with_columns( (pl.col('x_lat')/pl.col('norm_0')).alias('x_unit_0')  ,   (pl.col('y_lon')/pl.col('norm_0')).alias('y_unit_0')    , (pl.col('z_alt_n')/pl.col('norm_0')).alias('z_unit_0')          ) #Gets all points to be unit vectors. 
final_data = final_data.with_columns(  (   (pl.col('x_unit_0')**2 + pl.col('y_unit_0')**2 + pl.col('z_unit_0')**2 - 1) < var_eps    ).alias('is_on_sphere_0')   ) #Collection of indicators to see whether they are on the sphere. 
   
print(final_data.shape[0])
print("The number of registered flights was: ",n_f)
print("The number of error flights was:  ", n_e)
print("Number of points outside unit sphere", final_data.shape[0] - np.sum(final_data['is_on_sphere_0'].to_list()))
print()






#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


final_data = final_data.with_columns( (np.sin((np.pi/180)*pl.col('x_lat'))*np.cos((np.pi/180)*pl.col('y_lon'))).alias('x_sphr'),  (np.sin((np.pi/180)*pl.col('x_lat'))*np.sin((np.pi/180)*pl.col('y_lon'))).alias('y_sphr'),  (np.cos((np.pi/180)*pl.col('x_lat'))).alias('z_sphr'))
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


a=6378137
b = 6356752.314
final_data = final_data.with_columns( (pl.col('z_alt')*(0.3048)).alias('z_alt_m')   )
final_data = final_data.with_columns(  ( (a**2)/( np.sqrt(  (a*np.cos(pl.col('x_lat')*np.pi/180 ))**2   + (b*np.sin( pl.col('x_lat')*np.pi/180 ))**2  )  )   ).alias('N1') )
final_data = final_data.with_columns(    (  ( pl.col('N1') + pl.col('z_alt_m'))*np.cos(pl.col('x_lat')*np.pi/180)*np.cos(pl.col('y_lon')*np.pi/180)       ).alias('x_r0')  )
final_data = final_data.with_columns(    (  ( pl.col('N1') + pl.col('z_alt_m'))*np.cos(pl.col('x_lat')*np.pi/180)*np.sin(pl.col('y_lon')*np.pi/180)       ).alias('y_r0')  )
final_data = final_data.with_columns( (  (( (b/a)**2)*pl.col('N1')  + pl.col('z_alt_m') )*np.sin(pl.col('x_lat')*np.pi/180)  ).alias('z_r0')  )


final_data = final_data.with_columns(  (np.sqrt( pl.col('x_r0')**2 + pl.col('y_r0')**2 + pl.col('z_r0')**2   )).alias('norm_r0')        )
final_data = final_data.with_columns(     ( pl.col('x_r0')/pl.col('norm_r0')).alias('x_r0') , ( pl.col('y_r0')/pl.col('norm_r0')).alias('y_r0') , ( pl.col('z_r0')/pl.col('norm_r0')).alias('z_r0')      )
final_data = final_data.with_columns(  (   (pl.col('x_r0')**2 + pl.col('y_r0')**2 + pl.col('z_r0')**2 - 1) < var_eps    ).alias('is_on_sphere_r0')   )
print("Number of points outside unit sphere", final_data.shape[0] - np.sum(final_data['is_on_sphere_r0'].to_list()))
#Method 2, transformation. 
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

a=6378137
e=0.0818191908426

final_data = final_data.with_columns( (pl.col('x_lat')*np.pi/180).alias('x_lat_r'), (pl.col('y_lon')*np.pi/180).alias('y_lon_r') , (pl.col('z_alt')*np.pi/180*(0.3084)).alias('z_alt_r')  ) #Multiplies al coordinates by pi/180
final_data = final_data.with_columns(  (a/np.sqrt( 1- (np.sin(pl.col('x_lat_r')))**2) ).alias('N') )
final_data = final_data.with_columns(   (np.cos(pl.col('x_lat_r')) ).alias('cos_x_lat_r'), (np.cos(pl.col('y_lon_r')) ).alias('cos_y_lon_r')   )
final_data = final_data.with_columns(   (np.sin(pl.col('x_lat_r')) ).alias('sin_x_lat_r'), (np.sin(pl.col('y_lon_r')) ).alias('sin_y_lon_r')   )
final_data = final_data.with_columns(   ( (pl.col('N') + pl.col('z_alt_r'))*pl.col('cos_x_lat_r')*pl.col('cos_y_lon_r')  ).alias('x_act_r')   )
final_data = final_data.with_columns(   ( (pl.col('N') + pl.col('z_alt_r'))*pl.col('cos_x_lat_r')*pl.col('sin_y_lon_r')  ).alias('y_act_r')   )
final_data = final_data.with_columns(   ( (pl.col('N')*(1-e**2) + pl.col('z_alt_r'))*pl.col('sin_x_lat_r')  ).alias('z_act_r')   )



final_data = final_data.with_columns( ( (pl.col('x_act_r')**2 + pl.col('y_act_r')**2 + pl.col('z_act_r')**2)**(1/2)).alias('norm_vec'))
final_data = final_data.with_columns(  (pl.col('x_act_r')/pl.col('norm_vec')).alias('x_unit'), (pl.col('y_act_r')/pl.col('norm_vec')).alias('y_unit') , (pl.col('z_act_r')/pl.col('norm_vec')).alias('z_unit') )
final_data = final_data.with_columns(  (   (pl.col('x_unit')**2 + pl.col('y_unit')**2 + pl.col('z_unit')**2 - 1) < var_eps    ).alias('is_on_sphere')   )


#Method 2, transformation. 
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



#Gets difference between method 2 and method 3. 
a = 100*(final_data['x_unit'] - final_data['x_r0'])/final_data['x_r0']
print(a.describe())
b = (final_data['y_unit'] - final_data['y_r0']) #/final_data['y_unit']
print(b.describe())
c = 100*(final_data['z_unit'] - final_data['z_r0'])/final_data['z_r0']
print(c.describe())

print(final_data.shape[0])
print("The number of registered flights was: ",n_f)
print("The number of error flights was:  ", n_e)
print("Number of points outside unit sphere", final_data.shape[0] - np.sum(final_data['is_on_sphere'].to_list()))
plt.show()

print(final_data)
plt.hist(tim)
plt.show()


final_data.write_csv(save_file_name)


print(f"Were done! \n Data saved on: \n {save_file_name}")

