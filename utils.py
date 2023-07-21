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
    """Other options are: landsdel/landsdel.shp"""
    shapefile = "data/DAGI/" + shapefile
    regions_df = gp.read_file(shapefile)
    nodes_df = get_node_df(get_cleaned_graph())
    nodes_with_regions = gp.sjoin(nodes_df, regions_df, how="left", op="within")
    nodes_with_regions["node_id"] = nodes_with_regions["node_id"].astype(int)

    return nodes_with_regions

def get_municipality_flow():
    """ Combine trip information, stop coordinates, municipality shapefiles and election data
    to create a dataframe containing flow and distance between municipalities plus voting age population"""
    # Load required data
    df = pd.read_csv('data/jan_2019.csv', encoding = 'unicode_escape')
    stops = pd.read_csv('data/stops.txt')
    shapefile = "data/DAGI/kommuneinddeling/kommuneinddeling.shp"
    election_df = pd.read_csv('data/cleaned_election_data.csv')
    
    kommuner_df = gp.read_file(shapefile).to_crs('EPSG:4326') #convert to coordinates as in the stops df
    kommuner_df['kommunekod'] = kommuner_df['kommunekod'].astype(int)
    stops = gp.GeoDataFrame(stops, geometry=gp.points_from_xy(stops.stop_lon, stops.stop_lat), crs = 'EPSG:4326')
    stop_kommune = gp.sjoin(stops, kommuner_df, how="inner", op='intersects').set_index('stop_id')
    municipalities_df = stop_kommune['kommunekod']
    id2municipality = municipalities_df.to_dict()
    
    df['StartStopPointNr'] = df['StartStopPointNr'].map(id2municipality)
    df['SlutStopPointNr'] = df['SlutStopPointNr'].map(id2municipality)
    df = df.dropna()
    df['StartStopPointNr'] = df['StartStopPointNr'].astype(int)
    df['SlutStopPointNr'] = df['SlutStopPointNr'].astype(int)
    df['SUM_of_Personrejser'] = df['SUM_of_Personrejser'].astype(int)
    municipality_flow = df.drop('RejseUge', axis=1).groupby(['StartStopPointNr', 'SlutStopPointNr']).sum('SUM_of_Personrejser').reset_index()
    municipality_centroids = kommuner_df.to_crs('EPSG:3035').set_index('kommunekod').centroid #equal area projection
    m_centroid_dict = municipality_centroids.to_dict()
    
    municipality_flow = municipality_flow.rename(columns={"StartStopPointNr": "origin_id", "SlutStopPointNr": "destination_id", "SUM_of_Personrejser": "flow"})

    # Fill all municipality pairs not found with 0 flow
    ids = pd.unique(municipality_flow[['origin_id', 'destination_id']].values.ravel('K'))
    new_df = pd.DataFrame({'origin_id': np.repeat(ids, len(ids)), 'destination_id': np.tile(ids, len(ids))})
    # Merge new_df with df to get the corresponding flow values
    municipality_flow = pd.merge(new_df, municipality_flow, on=['origin_id', 'destination_id'], how='left')
    # Replace NaN values in the flow column with 0
    municipality_flow['flow'].fillna(0, inplace=True)
    municipality_population = election_df.groupby('Kommune').sum('Persons18')['Persons18'].reset_index().rename(columns = {'Persons18': 'population'})
    # Merge with origin
    df_combined = pd.merge(municipality_flow, municipality_population, how='left', 
                  left_on='origin_id', right_on='Kommune')
    df_combined = df_combined.rename(columns = {'population': 'origin_population'})
    df_combined = df_combined.drop(columns=['Kommune'])  # drop the extra Kommune column

    # Merge with destination
    df_combined = pd.merge(df_combined, municipality_population, how='left', 
                  left_on='destination_id', right_on='Kommune')
    df_combined = df_combined.rename(columns = {'population': 'destination_population'})
    df_combined = df_combined.drop(columns=['Kommune'])  # drop the extra Kommune column
    df_combined['flow'] = df_combined['flow'].astype(int)
    df_combined['distance'] = df_combined.apply(lambda row: m_centroid_dict[int(row['origin_id'])].distance(m_centroid_dict[int(row['destination_id'])]), axis=1)
    df_combined['origin_centroid'] = df_combined['origin_id'].map(m_centroid_dict)
    df_combined['destination_centroid'] = df_combined['destination_id'].map(m_centroid_dict)
    
    return df_combined

def get_election_df():
    df = pd.read_excel("data/cleaned_election_data.xlsx")
    df.drop(columns=["Unnamed: 0"], inplace=True)
    return df

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