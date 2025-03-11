import logging
import esdglider.config as config


"""
Scrape info from database, and generate deployment config file as yaml

'db/glider-db-prod.txt' is the database URL, used to create the 
sqlalchemy engine. It should not be committed to GitHub
"""

if __name__ == "__main__":
    logging.basicConfig(
        format='%(module)s:%(asctime)s:%(levelname)s:%(message)s [line %(lineno)d]', 
        level=logging.INFO, 
        datefmt='%Y-%m-%d %H:%M:%S')
    
    with open("db/glider-db-prod.txt", "r") as f:
        conn_string = f.read()

    config.make_deployment_config(
        "calanus-20241019", "ECOSWIM", "delayed", 
        "C:/SMW/Gliders_Moorings/Gliders/glider-lab/deployment-config", 
        conn_string)
