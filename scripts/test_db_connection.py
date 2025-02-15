# test mysql connection

import mysql.connector
from src.config.config import Config

mydb = mysql.connector.connect(
  host=Config.MYSQL_CONFIG["host"],
  user=Config.MYSQL_CONFIG["user"],
  password=Config.MYSQL_CONFIG["password"],
  database=Config.MYSQL_CONFIG["database"]
)

print(mydb)
