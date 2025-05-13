#Created by: Lars Daniel Johansson Niño
#Last edited date: 26/4/23
#Purpose: Perform python query tests with the traffic library 

"""
Some notes:
-   [DO NOT EDIT] The following might be necessary to activate the traffic library. 
    export PATH="$HOME/miniconda/bin:$PATH"
    conda activate traffic
-   [IMPORTANT] Using the traffic library requieres authentication, assuming the existance of an account on the Opensky 
    network, the user must edit the .conf (traffic.conf) file which comes with the installation of traffic
    by default. To access this file on Mac, one can search "traffic.conf" on Finder. 
-   [IMPORTANT] Accesing the Opensky network was done by changing the opensky.username and opensky.passwork atributes to their 
    correct values. The form provided by traffic doesnt always work.
-   [IMPORTANT] Opensky began having coverage for Mexico City around 2020-09-30
"""
import traffic 
import socket
import datetime
from traffic.data import opensky
from pathlib import PosixPath
import matplotlib.pyplot as plt

def main():
    print("Hello flight testing!\n \n")

    #Test provided by traffic 
    continues = False

    while continues == False:
        try:
            opensky.username = "Lars_Johansson57"
            opensky.password = "0se"


            #If the following works, well be able to get flight data for every 0.5 seconds
            data= opensky.history("2023-04-04 12:45",
            stop="2023-04-04 16:45", date_delta = datetime.timedelta(milliseconds=500),
            airport = "MMMX",
            )
            continues = True
            #The following is in 0.6 of a second
            #date_delta = datetime.timedelta(milliseconds=600)
            #date_delta = datetime.timedelta(milliseconds=60000), this works, but its by minute
            
            print(type(data))
        except socket.gaierror as e:
            continues = True
            print("Connection error! Saving data!\n",e)
        except:
            print("Possible authentication error! Execution continues")
main()



"""
Other notes:
-   [SOLVED] Currently the main problem regards activating the traffic library.
    The problem seems to relate to the installation of one of its dependencies, the Cartopy library

-   [SOLVED] There seems to be problems with the utilization of the Cartopy library, Im checking whether it 
    has to do with its dependencies. 

-   [ONGOING] We need a system to continuilly gather, calculate, and register the data we desire. 
    That is going to be some work. The code in this document tests some of the parts that might be useful
    for it. 

-   [ONGOING] I need to see whether I need to consider computer capabilities in the gathering of data. 
"""