import logging

import gspread
import sqlalchemy

import esdglider.config as config

"""
TODO

'db/glider-db-prod.txt' is the database URL, used to create the
sqlalchemy engine. It should not be committed to GitHub.
"""

# out_path = "C:/Users/sam.woodman/Downloads"
out_path = "../glider-lab-manual/content"

if __name__ == "__main__":
    logging.basicConfig(
        format="%(module)s:%(asctime)s:%(levelname)s:%(message)s [line %(lineno)d]",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    with open("db/glider-db-prod.txt", "r") as f:
        conn_string = f.read()
    engine = sqlalchemy.create_engine(conn_string)

    # Generate deployment table
    df_depl = config.make_deployment_table(engine)
    df_depl = df_depl.drop(["Dates", "Sensors"], axis=1)

    # df_depl.to_csv(
    #     "C:/Users/sam.woodman/Downloads/fleet-status-deployments.csv",
    #     index=False
    # )

    # Write Deployments table to fleet status
    wk_name = "Deployments-Database"
    logging.info("Updating the Fleet Status %s sheet", wk_name)
    df_depl = df_depl.fillna("").rename({"Glider_Deployment_ID": "Deployment_ID"})
    gc = gspread.oauth()
    sh = gc.open("Fleet Status")
    wk = sh.worksheet(wk_name)
    wk.update([df_depl.columns.values.tolist()] + df_depl.values.tolist())

    # # Update data validation formatting automatically..
    # wk.add_validation(
    #     f'F2:L{1+df_depl.shape[0]}',
    #     ValidationConditionType.one_of_list,
    #     ['TRUE', 'FALSE'],
    #     showCustomUi=True
    # )

    # Website yaml
    config.make_website_yaml(engine, out_path)
