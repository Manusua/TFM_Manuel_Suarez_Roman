""" Archivo auxiliar con funciones necesarias en la ejecución del notebook principal"""

import networkx as nx
import pandas as pd

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
def create_bipartite_graph(df, graphs_folder, manifestacion):
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
def create_graphs(node_criteria, edge_criteria, df, graphs_folder, manifestacion, filtered=False, thresh_filt=5):
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
