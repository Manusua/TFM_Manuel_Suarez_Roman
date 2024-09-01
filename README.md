# Autosimilitud en redes

El repositorio se encuentra organizado de la siguiente forma:

* ```Self similarity in graphs.ipynb```: notebook principal. Contiene todo el flujo de ejecuciones necesario para llegar a los resultados presentados en la memoria del Trabajo de Fin de Máster
* ```utils_graph.py```: funciones auxiliares del notebook de creación de grafo, obtención de métricas...
* ```nestedness_calculator.py``: implementa el algoritmo NODF para el calculo del coeficiente de anidamiento. Creado por  Mika Straka.
* ```datasets/```: directorio que contiene los conjuntos de datos obtenidos de Twitter de ambos movimientos sociales, No al Tarifazo (o nat) y 9 de noviembre (o 9n)
* ```measures/```: directorio que contiene las diferentes métricas calculadas para cada grafo, para evitar tener que recalcularlas.
* ```plots/```: contiene los diferentes gráficos generados en la ejecución del notebook principal.
* ```graphs/```: directorio con los diferentes grafos generados en formato ```.gexf```. Está estructurado de la siguiente forma:
        
        ├── graphs
        │    ├── bipartite: grafos bipartitos 
        |    │   ├── 9n: grafos de 9 de noviembre
        |    │   ├── nat: grafos de No al Tarifazo
        │    ├── nodes_hashtag: grafos con hashtags como nodos
        |    │   ├── 9n: grafos de 9 de noviembre
        |    │   ├── nat: grafos de No al Tarifazo
        │    ├── nodes_hashtag: grafos con usuarios como nodos
        |    │   ├── 9n: grafos de 9 de noviembre
        |    │   ├── nat: grafos de No al Tarifazo
    