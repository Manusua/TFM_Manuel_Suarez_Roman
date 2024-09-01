""" Archivo auxiliar con funciones necesarias en la ejecución del notebook principal"""

import networkx as nx
import pandas as pd
import os
import json
from tqdm import tqdm
import numpy as np


from nestedness_calculator import NestednessCalculator


########################################################################
#
# CREACION DE GRAFOS
#
########################################################################

# Lee el conjunto de datos y lo carga en un dataframe de pandas
def read_data(manifestacion, datasets_folder = "datasets/"):
    return pd.read_csv(datasets_folder + manifestacion + ".txt", sep= ' ')

# Crea, a partir de un umbral de filtro, thresh_filt, y de un grafo, un grafo nuevo eliminando las aristas del grafo original con peso inferior al umbral
def create_filtered_graph(G, thresh_filt):
    H = nx.Graph()
    # Añade nodos del grafo original al nuevo grafo
    H.add_nodes_from(G.nodes())

    # Añade aristas que cumplen con el umbral de peso
    for u, v, data in G.edges(data=True):
        if data['weight'] >= thresh_filt:
            H.add_edge(u, v, **data)
    return H

# Creación de grafos bipartitos dado un dataframe con la información de usuarios y hashtags de esa hora
def create_bipartite_graph(df, manifestacion, graphs_folder="graphs/"):
    graphs_folder = graphs_folder + 'bipartite/' + manifestacion + '/'
    df_h = df["hour"].unique()
    print("Creando redes bipartitas, manifestación seleccionada:", manifestacion, "número de horas: ", len(df_h))
    G = nx.Graph()
    for hour in df_h:
        df_hour = df[(df["hour"] == hour)]
        G = nx.from_pandas_edgelist(df_hour, source="user", target="hashtag", edge_attr="weight")
        nx.write_gexf(G, graphs_folder + str(hour) + ".gexf")

# Crea grafos dado un dataframe con la información de usuarios y hashtags de esa hora atendiendo al node_criteria, 
# que determina si los nodos de las redes son hashtags o usuarios para la manifestación determinada por parámetros.
# Tiene una opción de crear grafos eliminando las aristas con peso inferior a thresh_filt
def create_graphs(node_criteria, edge_criteria, df, manifestacion, graphs_folder="graphs/", filtered=False, thresh_filt=5):
    graphs_folder = graphs_folder + 'nodes_' + node_criteria + '/'+ manifestacion + '/'
    df_h = df["hour"].unique()
    print("Creando redes de", node_criteria, "unidos si comparten uno o más", edge_criteria, ", manifestación seleccionada:", manifestacion, "número de horas: ", len(df_h))
    for hour in df_h:
        G = nx.Graph()
        df_hour = df[(df["hour"] == hour)]
        df_nodes = df_hour[node_criteria].unique()
        G.add_nodes_from(df_nodes)
        for node in df_nodes:
            # Seleccionamos las filas del dataframe con el usuario/hashtag sobre el que iteramos
            df_node_edge = df_hour.loc[df_hour[node_criteria] == node]
            # Seleccionamos tantos hashtags/usuarios como haya que haya compartido el usuario/hasthag respectivamente
            df_node_edge = df_node_edge[edge_criteria]
            for edge in df_node_edge:
                df_edge = df_hour.loc[df_hour[edge_criteria] == edge]
                df_edge = df_edge[node_criteria]
                for nd in df_edge:
                    if nd != node:
                        if G.has_edge(node, nd):
                            G[node][nd]["weight"] += 1
                        else:
                            G.add_edge(node, nd, weight = 1)
        
        # Finalmente dividimos entre dos todos los pesos de las aristas, pues están contados dos veces (uno por cada nodo)
        for edge in G.edges():
            old_weight = G.edges[edge]["weight"]
            nx.set_edge_attributes(G, {edge: {"weight": old_weight/2}})

        if filtered:
            G = create_filtered_graph()
            graphs_folder = graphs_folder + "filtered/" + str(thresh_filt) + '/'

        nx.write_gexf(G, graphs_folder + str(hour) + ".gexf")


########################################################################
#
# OBTENCIÓN DE MÉTRICAS
#
########################################################################


# Función para convertir las claves de cadenas a enteros o decimales (necesaria al leer de un json)
def convert_keys_to_float(d, recursive=True, tipo="float"):
    new_dict = {}
    for k, v in d.items():
        # Convierte la clave a entero o a float si es posible
        try:
            if tipo == "int":
                k = int(k)
            else:
                k = float(k)
        except ValueError:
            pass
        
        if recursive:
            # Si el valor es un diccionario, aplica la conversión recursivamente
            if isinstance(v, dict):
                v = convert_keys_to_float(v)
        
        new_dict[k] = v
    return new_dict

# Se importa el nestedness calculator que implementa NODF
def calc_nestedness(G):
    mat = nx.to_numpy_array(G, weight=None)
    mat = mat[~np.all(mat == 0, axis=1)]
    mat = mat[:,~np.all(mat == 0, axis=0)]
    nodf_score = NestednessCalculator(mat).nodf(mat)
    return nodf_score

# Recibe como paarámetros, la manifestación que se está tratando y el modo de la misma ("h" para redes con hashtags como nodos, 
# 'u' para redes con usuarios como nodos y 'b' para redes bipartitas). Devuelve las horas de la manifestación ordenadas, con
# el coeficiente de clustering y de anidamiento de cada una
def get_clust_nest_coefficient(manifestacion, criterio, measures_foler="measures/", datasets_foler="datasets/", graphs_folder="graphs/", write=True, read=True):

    print("Calculando el anidamiento y modularidad de " + manifestacion + " con criterio: " + criterio)
    if criterio == "h":
        name_path = "hashtag"
    elif criterio == "u":
        name_path = "user"

    dict_manif = {}
    path_file = measures_foler + manifestacion + '_' + criterio + '.json'

    if read:
    # Intentamos cargar el archivo que contenga los datos (si existe) si está activa la flag de lectura
        if os.path.exists(path_file):
            try:
                with open(path_file) as f:
                    dict_manif = json.load(f)
                dict_manif = convert_keys_to_float(dict_manif, recursive=False, tipo="int")
            except json.JSONDecodeError:
                dict_manif = {}

    df = read_data(manifestacion, datasets_folder=datasets_foler)
    horas = df["hour"].unique()

    # Se ve que infomación del grafo está ya calculada y, si no, se calcula
    for hora in tqdm(horas):
        hora = int(hora)

        if not hora in dict_manif.keys():
            dict_manif[hora] = {}

        if not ("nestedness" in dict_manif[hora].keys() and "modularity" in dict_manif[hora].keys()):
            if not criterio is "b":
                G = nx.read_gexf(graphs_folder + 'nodes_' + name_path + '/' + manifestacion + '/' + str(hora) + '.gexf')
            else:
                G = nx.read_gexf(graphs_folder + 'bipartite/' + manifestacion + '/' + str(hora) + '.gexf')
            if not "nestedness" in dict_manif[hora].keys():
                nestedness = calc_nestedness(G)
                dict_manif[hora]["nestedness"] = float(nestedness)
            
            if not "modularity" in dict_manif[hora].keys(): 
                modularity_louv = nx.community.modularity(G, nx.community.louvain_communities(G, seed=123), weight="weight")
                dict_manif[hora]["modularity"] = modularity_louv

    arr_hour =[]
    arr_nest = []
    arr_mod = []

    for k in dict_manif.keys():
        arr_hour.append(str(k))
        arr_nest.append(dict_manif[k]["nestedness"])
        arr_mod.append(dict_manif[k]["modularity"])

    # Se ordena la información de menor a mayor hora
    data = list(zip(arr_hour, arr_mod, arr_nest))
    data.sort()
    hour_sort, mod_sort, nest_sort = zip(*data)
    hour_sort = list(hour_sort)
    mod_sort = list(mod_sort)
    nest_sort = list(nest_sort)

    # Si esta activa la flag de escritura, se guarda la información en un archivo para no tener que recalular en un futuro
    if write:
        with open(path_file, 'w') as f:
            json.dump(dict_manif, f, indent=2)

    return hour_sort, mod_sort, nest_sort


########################################################################
#
# OBTENCIÓN DE COEFICIENTES DE LEY DE POTENCIA
#
########################################################################