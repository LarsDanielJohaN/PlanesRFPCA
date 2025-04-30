"""
Created by: Lars Daniel Johansson Niño
Date:2/5/23
Purpose: Create first version of code to generate observations
"""
import pandas as pd
import traffic
from traffic.data import opensky
import socket 
import datetime
import pyproj as pypr
import geopy
from geopy.distance import geodesic
from shapely.geometry import Polygon, Point



class Test_Gen_Obs:
    def __init__(self,airport: str, time_breaks:int , user:str, password:str, start_time:str, end_time:str, points:pd.DataFrame,reference_polygon:Polygon):
        #General class attributes required for Opensky´s queries
        self.user = user
        self.airport = airport
        self.time_breaks = datetime.timedelta(milliseconds=time_breaks)
        self.password = password
        self.start_time = start_time
        self.end_time = end_time
        self.points = points
        self.referece_polygon = reference_polygon
        self.error = False
 

    #This method makes the query request to Opensky for flights in a particular
    #timestamp. 


    """
    I might need to modify the request method to use Impala.request, this makes a more primitive requesto to
    Opensky, but it allows to set airspace bounds, which can be usefull for our purposes. 
    """
    def make_request(self) -> pd.DataFrame:
        continues = False
        data = None

        while continues == False:
            try:
                opensky.username = self.user
                opensky.password = self.password
                data= opensky.history(self.start_time,
                stop= self.end_time, date_delta = self.time_breaks,
                airport = self.airport,
                )
                """
                data = opensky.extended(self.start_time, stop = self.end_time,
                date_delta = self.time_breaks, airport = self.airport, bounds = self.bounds)
                con bounds siendo un espacio aereo o un poligono para establecer los limites
                """
                continues = True

            #Catches connection error
            except socket.gaierror as e:
                print("Connection error!\n",e)
                self.error = True
                continues = True

            #Catches error of any other kind
            except:
                continues = False
                print("Possible authentication error! Execution continues")
        return data

    #This method processes flights in desired timestamp

    

#The use of the variable data is temporary, the intention is to fully automatize everything
    def handle_flights(self, data:pd.DataFrame) -> pd.DataFrame:
        print("Hello flight handler!")

        """
        'Unnamed: 0.1', 'Unnamed: 0', 'alert', 'altitude', 'callsign', 'day',
       'destination', 'firstseen', 'geoaltitude', 'groundspeed', 'hour',
       'icao24', 'last_position', 'lastseen', 'latitude', 'longitude',
       'onground', 'origin', 'spi', 'squawk', 'timestamp', 'track',
       'vertical_rate'
        """
        
        print(data.columns)
        print(self.points.columns)

        for point in self.points.index:
            curr = self.points['CVEGEO'][point]


        #for point in data.index:
         #   distances = []
          #  curr_flight = (data['longitude'][point], data['latitude'][point])


            #curr_Loc = 
            #dist = GD(curr_flight,curr_Loc).m
            #print(data['longitude'][point], data['latitude'][point])

    def calculate_geodesic_distance(a:Point, b:Point):
        print("Hello, this will calculate the distance between two points")

        

        

        

            


        
     

        #This calculates the distance from an airplane to a
        #point in meters. 
        
        
    



       





