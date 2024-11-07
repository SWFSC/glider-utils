# 

import os
import pandas as pd
# import pyodbc
import sqlalchemy 


conn_path = os.path.join(
    "C:/SMW/Gliders_Moorings/Gliders/esd-glider-utils", 
    "db", "glider-db-prod.txt"
)

with open(conn_path, "r") as f:
    conn_string = f.read()
# print(conn_string)

# establish connection with the database 
engine = sqlalchemy.create_engine(conn_string)
print(engine)

# conn = pyodbc.connect(dsn_string)
# print(conn)

# print(pd.read_sql('SELECT * FROM Glider_Deployment', engine))
print(pd.read_sql_table('Glider_Deployment', con = engine, schema = 'dbo'))
print(pd.read_sql_table('vGlider', con = engine, schema = 'dbo'))
