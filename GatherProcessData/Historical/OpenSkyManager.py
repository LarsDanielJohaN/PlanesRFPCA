"""
Author: Lars Daniel Johansson Niño
Last edited: 25/07/24
Purpose: Communicate with OpenSky
"""
from sqlalchemy import create_engine
from sqlalchemy.schema import Table, MetaData
from sqlalchemy.sql.expression import select, text


class OpenSkyManager:


    def __init__(self, user:str, password:str):
        print("Were onnnnnnnnn")
        self.usr = user 
        self.pswd = password
        url = f"trino://'{self.usr}':'{self.pswd}'@trino.opensky-network.org:443/mimio/osky"
        print(url)
        print()
        print()
        self.eng = create_engine(url)
        self.conn = self.eng.connect()

        print(self.conn.execute(text("show tables;")).fetchall() )
