import pandas as pd # used to load individual files
import numpy as np # not currently used
import datetime as dt # not currently used
#from geopy.distance import geodesic might be used at some point
#import peartree as pt # used to construct network
import matplotlib.pyplot as plt
import osmnx as ox # visualisation library for networks
import pickle
import networkx as nx
import geopandas as gp
from collections import defaultdict
from geopy.distance import geodesic
from tqdm import tqdm
from sklearn.neighbors import BallTree
import math


def get_cleaned_graph():
    with open('rejsekort_graph_cleaned.gpickle', 'rb') as f:
           G = pickle.load(f)
    return G

def get_node_df(G):
    node_attributes = {
    "node_id":list(G.nodes()),
    "longitude":[G.nodes[node]['x'] for node in G.nodes()],
    "latitude":[G.nodes[node]['y'] for node in G.nodes()],
    "modes":[G.nodes[node]['modes'] for node in G.nodes()],
}
    nodes_df = gp.GeoDataFrame.from_dict(node_attributes)
    nodes_df.geometry = gp.points_from_xy(nodes_df.longitude, nodes_df.latitude)
    nodes_df.crs = "EPSG:4326" # discovered from trial and error
    nodes_df.to_crs("EPSG:25832", inplace=True) # crs of rejsekort data
    return nodes_df

def get_nodes_with_regions_df(shapefile="kommuneinddeling/kommuneinddeling.shp"):
    regions_df = gp.read_file(shapefile)
    nodes_df = get_node_df(get_cleaned_graph())
    nodes_with_regions = gp.sjoin(nodes_df, regions_df, how="left", op="within")
    nodes_with_regions["node_id"] = nodes_with_regions["node_id"].astype(int)

    return nodes_with_regions



KOMMUNE_CODE_NAME_DICT = {
    "0101": "København",
    "0147": "Frederiksberg",
    "0151": "Ballerup",
    "0153": "Brøndby",
    "0155": "Dragør",
    "0157": "Gentofte",
    "0159": "Gladsaxe",
    "0161": "Glostrup",
    "0163": "Herlev",
    "0165": "Albertslund",
    "0167": "Hvidovre",
    "0169": "Høje-Taastrup",
    "0173": "Lyngby-Taarbæk",
    "0175": "Rødovre",
    "0183": "Ishøj",
    "0185": "Tårnby",
    "0187": "Vallensbæk",
    "0190": "Furesø",
    "0201": "Allerød",
    "0210": "Fredensborg",
    "0217": "Helsingør",
    "0219": "Hillerød",
    "0223": "Hørsholm",
    "0230": "Rudersdal",
    "0240": "Egedal",
    "0250": "Frederikssund",
    "0253": "Greve",
    "0259": "Køge",
    "0260": "Halsnæs",
    "0265": "Roskilde",
    "0269": "Solrød",
    "0270": "Gribskov",
    "0306": "Odsherred",
    "0316": "Holbæk",
    "0320": "Faxe",
    "0326": "Kalundborg",
    "0329": "Ringsted",
    "0330": "Slagelse",
    "0336": "Stevns",
    "0340": "Sorø",
    "0350": "Lejre",
    "0360": "Lolland",
    "0370": "Næstved",
    "0376": "Guldborgsund",
    "0390": "Vordingborg",
    "0400": "Bornholm",
    "0410": "Middelfart",
    "0411": "Christiansø",
    "0420": "Assens",
    "0430": "Faaborg-Midtfyn",
    "0440": "Kerteminde",
    "0450": "Nyborg",
    "0461": "Odense",
    "0479": "Svendborg",
    "0480": "Nordfyns",
    "0482": "Langeland",
    "0492": "Ærø",
    "0510": "Haderslev",
    "0530": "Billund",
    "0540": "Sønderborg",
    "0550": "Tønder",
    "0561": "Esbjerg",
    "0563": "Fanø",
    "0573": "Varde",
    "0575": "Vejen",
    "0580": "Aabenraa",
    "0607": "Fredericia",
    "0615": "Horsens",
    "0621": "Kolding",
    "0630": "Vejle",
    "0657": "Herning",
    "0661": "Holstebro",
    "0665": "Lemvig",
    "0671": "Struer",
    "0706": "Syddjurs",
    "0707": "Norddjurs",
    "0710": "Favrskov",
    "0727": "Odder",
    "0730": "Randers",
    "0740": "Silkeborg",
    "0741": "Samsø",
    "0746": "Skanderborg",
    "0751": "Aarhus",
    "0756": "Ikast-Brande",
    "0760": "Ringkøbing-Skjern",
    "0766": "Hedensted",
    "0773": "Morsø",
    "0779": "Skive",
    "0787": "Thisted",
    "0791": "Viborg",
    "0810": "Brønderslev",
    "0813": "Frederikshavn",
    "0820": "Vesthimmerlands",
    "0825": "Læsø",
    "0840": "Rebild",
    "0846": "Mariagerfjord",
    "0849": "Jammerbugt",
    "0851": "Aalborg",
    "0860": "Hjørring"
}