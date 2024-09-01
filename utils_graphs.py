""" Archivo auxiliar con funciones necesarias en la ejecución del notebook principal"""

import networkx as nx
import pandas as pd
import os
import json
from tqdm import tqdm
import numpy as np
import powerlaw


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


# Calcula la media de grados de un grafo
def calc_avg_degree(G):
    return sum(dict(G.degree).values())/G.number_of_nodes()

# Función que añade los nodos a un grafo dado un threshold
def add_nodes_subgraph(G, threshold):
    F = nx.Graph()
    for node in G.nodes():
        # Se comprueba si el grado del nodo es mayor que el umbral y se añade
        if G.degree[node] > threshold:
            F.add_node(node)
    return F

# Función que dado un grafo G y un subgrafo suyo, F, solo con nodos, añade las aristas a F
# si dos nodos de F tienen arista en G
def add_edges_subgraph(G, F):
    for node in F.nodes():
        # Se itera sobre los vecinos en G de cada nodo y se ve si pertenecen a F
        for neighbor in G.neighbors(node):
            if neighbor in F.nodes():
                # Se añade la arista si no existe ya
                if not neighbor in F.neighbors(node):
                    F.add_edge(node, neighbor)
    return F


# Se añade la variable "internalDegree" a cada nodo dada la media de grados del subgrafo
# y el grado del propio nodo
def add_hidden_variable(F):
    avg_deg = calc_avg_degree(F)
    if avg_deg != 0:
        dict_hidd_var = {}
        for node in F.nodes():
            dict_hidd_var[node] = F.degree[node] / avg_deg
        nx.set_node_attributes(F, dict_hidd_var, "internalDegree")
    else:
        return -1

# Dado un grafo original y un umbral, aplica el proceso de normalización por umbral de grado descrito en el artículo 
# "Self-similarity of complex networks and hidden metric spaces" de Angeles et al.
def thresh_normalization(G, threshold):

    # Añadimos solamente los nodos que cumplan el umbral
    F = add_nodes_subgraph(G, threshold)
    
    # Si ya no hay nodos que cumplan el umbral, se acaba el proceso
    if F.number_of_nodes() == 0:
        return -1
    
    # Ahora se añaden las aristas de G de los nodos en el subgrafo F
    F = add_edges_subgraph(G, F)

    # Se añade como variable oculta el grado entre la media del grafo a cada nodo
    if add_hidden_variable(F) == -1:
        return -1
    
    return F


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
            if criterio != "b":
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



# Dado un grafo y un diccionario con la información clust[nodo] = coeficiente de clusterizacion del nodo
# devuelve un diccionario con internal degrees normalizados como clave y la media de coeficiente de clusterización
# de los nodos que tienen dicho internal degree como valor
def calc_avg_clust_coef_by_normalized_internal_degree(G, clust):
    dict_hid_var_aux = {}
    dict_hid_var = {}
    # Crea un diccionario con cada internal degree como clave y un array con los coeficientes
    # de clusterización de los nodos que tienen dicho internal degree
    arr_int_deg = []

    for node in G.nodes():
        att = G.nodes[node]["internalDegree"]
        if att in dict_hid_var_aux.keys():
            np.append(dict_hid_var_aux[att], clust[node])
        else:
            dict_hid_var_aux[att] = np.array(clust[node])

        arr_int_deg.append(att)

    # Se ordena el diccionario en función de la clave (internal degree) de menor a mayor
    # sorted(dict) devuleve las keys ordenadas
    dict_hid_var_aux_2 = {k: dict_hid_var_aux[k] for k in sorted(dict_hid_var_aux)}

    # Se crea un diccionario con internal degrees como clave y la media de coeficiente de clusterización
    # de los nodos que tienen dicho internal degree como valor
    for key in dict_hid_var_aux_2.keys():
        dict_hid_var[key] = np.average(dict_hid_var_aux_2[key])

    return dict_hid_var

# Crea MAX_UMBRAL sucesivos subgrafos tras aplicar iterativamene el proceso de normalización
# y calcula la distribución de grados y devuelve 
#   - dict_thres_avg_clust: diccionario con thresholds como claves 
#       y media de coeficientes de clusterización del subgrafo generado con dicho threshold como valor
#   - arr_norm_int_deg_fig2a: array con diccionarios con internal degrees como clave y la media de coeficiente de clusterización
#       de los nodos que tienen dicho internal degree como valor. Cada índice se corresponde con el threshold empleado
#       para generar el subgrafo
def calc_clust(G, MAX_UMBRAL, measures_path, mode="h", read=True, write=True):
    
    dict_thres_avg_clust = {}
    dict_norm_int_deg = {}

    measures_path = measures_path + '_' + mode
    if read:
        # Se intenta cargar el archivo donde están los datos
        if os.path.exists(measures_path + '_int_deg.json'):
            try:
                with open(measures_path + '_int_deg.json', 'r') as f:
                    dict_norm_int_deg = json.load(f)
                dict_norm_int_deg = convert_keys_to_float(dict_norm_int_deg)
            except json.JSONDecodeError:
                dict_norm_int_deg = {}

        if os.path.exists(measures_path + '_avg_clust.json'):
            try:
                with open(measures_path + '_avg_clust.json', 'r') as f:
                    dict_thres_avg_clust = json.load(f)
                dict_thres_avg_clust = convert_keys_to_float(dict_thres_avg_clust)
            except json.JSONDecodeError:
                dict_thres_avg_clust = {}

    # Se calculan las sucesivas métricas para cada umbral de grado y se escriben
    for threshold in tqdm(range(MAX_UMBRAL)):
        flag_int_deg = threshold in dict_norm_int_deg.keys()
        flag_avg_clust = threshold in dict_thres_avg_clust.keys()
        # Si ambas son true podemos saltar el paso (ya está calculado previamente)
        if not (flag_int_deg and flag_avg_clust):
            # Se crea el subgrafo basándonos en el threshold seleccionado
            F = thresh_normalization(G, threshold)
            if F == -1:
                # Caso de grafo vacío o grafo inconexo
                # Se escribe la información calculada.
                if write:
                    with open(measures_path + '_int_deg.json', "w") as f:
                        json.dump(dict_norm_int_deg, f, indent=2)
                    with open(measures_path + '_avg_clust.json', "w") as f:
                        json.dump(dict_thres_avg_clust, f, indent=2)

                return dict_thres_avg_clust, dict_norm_int_deg

            if mode == "b":
                clust = nx.algorithms.bipartite.clustering(F)
            else:
                clust  = nx.clustering(F)
            avg_clust = np.mean(np.array(list(clust.values())))
            dict_thres_avg_clust[threshold] = avg_clust

            dict_norm_int_deg[threshold] = calc_avg_clust_coef_by_normalized_internal_degree(F, clust)
        if write:
            with open(measures_path + '_int_deg.json', "w") as f:
                json.dump(dict_norm_int_deg, f, indent=2)
            with open(measures_path + '_avg_clust.json', "w") as f:
                json.dump(dict_thres_avg_clust, f, indent=2)
    return dict_thres_avg_clust, dict_norm_int_deg

# Carga el grafo demandado por parámetros y devuelve los diccionarios con las métricas de autosimilitud
def calc_self_sim(hora, MAX_UMBRAL, manifestacion, mode='h', graphs_folder="graphs/", measures_folder="measures/"):
    if mode == "h":
        path_graph = "nodes_hashtag/"
    elif mode == "u":
        path_graph = "nodes_user/"
    elif mode == "b":
        path_graph = "bipartite/"
    G = nx.read_gexf(graphs_folder + path_graph  + manifestacion + hora + ".gexf")
    path_measures_hour = measures_folder + manifestacion
    if not os.path.exists(path_measures_hour):
        os.makedirs(path_measures_hour)
    return calc_clust(G, MAX_UMBRAL, path_measures_hour + hora, mode=mode)

########################################################################
#
# OBTENCIÓN DE COEFICIENTES DE LEY DE POTENCIA
#
########################################################################

# Recibe como parámetros de array de puntos, correpsondiente con los grados de los nodos de un grafo.
# Devuelve un array con la información de los exponentes hallados para cada elemento
def get_exp(arr_points, name_graph, measures_folder="measures/"):
    
    measures_path = measures_folder + "degrees/" 
    # Se ordenan los puntos de menor a mayor quitando los 0s (producen error al calcular el exponente)
    points_aux = np.sort(arr_points)
    points_aux = points_aux[points_aux != 0]
    points_aux = points_aux[::-1]

    """    # Se escriben los grados de los nodos en un archivo.
    # El formato es este para poder emplear d-mercator
    path = measures_path + name_graph + '.txt'
    with open(path, "w") as f:
        for point in points_aux:
            f.writelines(str(point) + '\n')"""

    results = powerlaw.Fit(points_aux)
        
    return results

# Dado un array de arrays correspondiente con los grados (o los grados normalizados) de un grafo,
# devuelve un array de tuplas.Cada tupla tiene como primer elementos los grados de los nodos (eje X)
# y las probabilidades de que cada nodo tenga dicho grado (o grado normalizado) (eje Y) como segundo elemento.
def calc_pdf_points(arr_points):
    arr_pdf_points = []
    for points in arr_points:
        degrees, counts = np.unique(points, return_counts=True)
        probs = counts / len(points)
        arr_pdf_points.append((degrees, probs))
    return arr_pdf_points

# Dado un array de arrays correspondiente con los grados de los nodos (eje X) y las probabilidades de que cada nodo tenga dicho grado (PDF)
# devuelve un array de tuplas donde el primer elemento de cada tupla sigue siendo el grado de los nodos y el segundo es la probabilidad
# cumulativa de que un nodo tenga dicho grado.
def calc_cdf_points(arr_pdf_points):
    arr_cdf_points = []
    for pdf_points in arr_pdf_points:
        cdf = np.cumsum(pdf_points[1])
        arr_cdf_points.append((pdf_points[0], cdf))
    return arr_cdf_points

# Dado un array de arrays correspondiente con los grados de los nodos (eje X) y las probabilidades cumulativa de que un nodo tenga dicho grado (CDF)
# devuelve un array de tuplas donde el primer elemento de cada tupla sigue siendo el grado de los nodos y el segundo es la probabilidad
# cumulativa complementaria de que un nodo tenga dicho grado.
def calc_ccdf_points(arr_cdf_points):
    arr_ccdf_points = []
    for deg_cum in arr_cdf_points:
        ccdf = 1 - deg_cum[1]
        # Se quita el último punto pues, al ser escala logaritimica en los ejes y ser su probablidad
        # complementaria 0 o muy cercana a 0, hace que el grafico quede deformado y no es util
        arr_ccdf_points.append((deg_cum[0][:-1], ccdf[:-1]))   
    return arr_ccdf_points

# Dado un grafo, calcula la distgribución de grados de sus nodos, así como la PDF, CDF y CCDF de la probabilidad de los grados. Además también calcula el exponente del ajuste
# de la ley de potencia a la distribución si exp=True
def calc_degree_distribution(hour, manifestacion, graphs_folder="graphs/", mode="h", measures_folder="measures/", G=None, arr_kt=[0], exp=False, read=True, write=True):
    if mode == "h":
        path_graph = "nodes_hashtag/"
    elif mode == "u":
        path_graph = "nodes_user/"
    elif mode == "b":
        path_graph = "bipartite/"

    # Se carga el grafo inicial
    G = nx.read_gexf(graphs_folder + path_graph + manifestacion + hour + '.gexf')
        
    arr_points = []
    dict_points = {}
    measures_path = measures_folder + manifestacion + hour + '_' + mode + '_degs_kt.json'
    if read:
        # En los archivos se guardan los grados de los nodos de cada grafo dependiente de kt en bruto, sin sufrir procesos de normalización
        if os.path.exists(measures_path):
            try:
                with open(measures_path, 'r') as f:
                    dict_points = json.load(f)
                dict_points = convert_keys_to_float(dict_points, tipo="int")
            except json.JSONDecodeError:
                pass
    
    # Si no recibe un arr_kt como parametro, se interpreta que es la red original
    for kt in tqdm(arr_kt):
        if not kt in dict_points.keys():
            F = thresh_normalization(G, kt)
            if F != -1:
                points_kt = np.sort(np.array(list(dict(F.degree()).values())).astype(float))
                dict_points[kt] = list(points_kt)
    
        else:
            points_kt = dict_points[kt]
        points_kt = np.array(points_kt) / np.mean(points_kt)

        arr_points.append(points_kt)
    
    # El exponente solo se va a calcular cuando se reciba un valor de kt
    plfit = None
    if exp:
        # Para calcular el exponente no hay que normalizar
        plfit = get_exp([points_kt*np.mean(points_kt)], hour)

    # Puntos de la PDF
    arr_deg_prob = []
    for points in arr_points:
        degrees, counts = np.unique(points, return_counts=True)
        probs = counts / len(points)
        arr_deg_prob.append((degrees, probs))
    
    # Puntos de la CDF
    arr_deg_cum = []
    for deg_prob in arr_deg_prob:
        cum_freq = np.cumsum(deg_prob[1])
        cdf = cum_freq/cum_freq[-1]
        arr_deg_cum.append((deg_prob[0], cdf))

    # Puntos de la CCDF
    arr_deg_comp_cum = []
    for deg_cum in arr_deg_cum:
        ccdf = 1 - deg_cum[1]
        arr_deg_comp_cum.append((deg_cum[0], ccdf))

    if write:
        with open(measures_path, "w") as f:
            json.dump(dict_points, f, indent=2)
    return plfit, arr_deg_prob, arr_deg_comp_cum

########################################################################
#
# FUNCIONES DE REPRESENTACIÓN DE MÉTRICAS
#
########################################################################

# Para seleccionar diferentes markers al representar gráficamente
def get_all_markers():
    return [
    '.',  # point marker
    ',',  # pixel marker
    'o',  # circle marker
    'v',  # triangle_down marker
    '^',  # triangle_up marker
    '<',  # triangle_left marker
    '>',  # triangle_right marker
    '1',  # tri_down marker
    '2',  # tri_up marker
    '3',  # tri_left marker
    '4',  # tri_right marker
    's',  # square marker
    'p',  # pentagon marker
    '*',  # star marker
    'h',  # hexagon1 marker
    'H',  # hexagon2 marker
    '+',  # plus marker
    'x',  # x marker
    'D',  # diamond marker
    'd',  # thin_diamond marker
    '|',  # vline marker
    '_',  # hline marker
    'P',  # plus (filled) marker
    'X',  # x (filled) marker
    0,    # tickleft marker
    1,    # tickright marker
    2,    # tickup marker
    3,    # tickdown marker
    4,    # caretleft marker
    5,    # caretright marker
    6,    # caretup marker
    7    # caretdown marker
]