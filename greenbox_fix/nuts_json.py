
from nuts_finder import NutsFinder
import numpy as np

nuts = NutsFinder(year=2013, scale=1)
nuts_results = {}
for i in range(len(nuts.shapes['features'])):
    region = nuts.shapes['features'][i]
    log_msg(i, region["properties"]["LEVL_CODE"], region['id'])
    if region["properties"]["LEVL_CODE"] == 2:
        try:
            nuts_results[region["id"]] = np.average(np.asarray(region["geometry"]["coordinates"]).flatten().reshape(-1,2),axis=0).tolist()
        except:
            try:
                nuts_results[region["id"]] = np.average(np.asarray(region["geometry"]["coordinates"][0]).flatten().reshape(-1,2),axis=0).tolist()
            except:
                nuts_results[region["id"]] = np.average(np.asarray(region["geometry"]["coordinates"][0][0]).flatten().reshape(-1,2),axis=0).tolist()

log_msg(nuts_results)
exit(0)