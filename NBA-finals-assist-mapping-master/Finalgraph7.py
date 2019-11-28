import networkx as nx
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import itertools as it

def readfile(filename):
    with open(filename) as r:
        dataLines = r.read().splitlines()

    data = []
    for z in range(0, len(dataLines)):
        data.append(dataLines[z].split(','))
        print (data[z-1])

    tags = []
    for i in range(len(data)):
        temp = []
        for j in range(0, len(data[0])):
            if data[i][j] != '' and data[i][j] != '\r\n':
                temp.append(str(data[i][j]))

        tags.append(temp)

    graph = []
    for i in range(len(tags)):
        temp_tags = list(it.combinations(tags[i], 2))
        for n in temp_tags:
            graph.append(n)

    return (graph)

def drawgraph(graph, node_color, edge_color, title, labels=None, node_size=400, node_alpha=0.4, node_text_size=12, edge_alpha=0.3, edge_text_pos=0.3, text_font='Arial'):
    G=nx.MultiGraph()


    for edge in graph:
        G.add_edge(edge[0], edge[1], attr_dict=None, weight=7)

    edgewidth=[]
    for (u,v,d) in G.edges(data=True):
        edgewidth.append(7*len(G.get_edge_data(u,v)))

    nodesize = []
    for (u,v,d) in G.edges(data=True):
        nodesize.append(u)
        nodesize.append(v)

    bigball = [nodesize.count(v)*750 for v in G]
    graph_pos=nx.shell_layout(G)

    nx.draw_networkx_nodes(G,graph_pos,node_size=bigball,
                           alpha=node_alpha, node_color=node_color)
    nx.draw_networkx_edges(G,graph_pos,width=edgewidth,
                           alpha=edge_alpha,edge_color=edge_color)
    nx.draw_networkx_labels(G, graph_pos,font_size=node_text_size,
                            font_family=text_font, font_weight='bold')  
    labels = ''

    edge_labels = dict(zip(graph, labels))
    nx.draw_networkx_edge_labels(G, graph_pos, edge_labels=edge_labels,
                                 label_pos=edge_text_pos, font_weight ='normal', alpha = 1.0,fontsize = 10)

    font = {'fontname': 'Arial',
            'color': 'y',
            'fontweight': 'bold',
            'fontsize': 19}
    plt.title(title, font)
    plt.axis('off')
    plt.show()


graph1 = readfile("/Users/xiaoy/Downloads/NBA-finals-assist-mapping-master/dubsG1asts.csv")
graph2 = readfile("/Users/xiaoy/Downloads/NBA-finals-assist-mapping-master/cavsG1asts.csv")

graph3 = readfile("/Users/xiaoy/Downloads/NBA-finals-assist-mapping-master/dubsG2asts.csv")
graph4 = readfile("/Users/xiaoy/Downloads/NBA-finals-assist-mapping-master/cavsG2asts.csv")

graph5 = readfile("/Users/xiaoy/Downloads/NBA-finals-assist-mapping-master/dubsG3asts.csv")
graph6 = readfile("/Users/xiaoy/Downloads/NBA-finals-assist-mapping-master/cavsG3asts.csv")

graph7 = readfile("/Users/xiaoy/Downloads/NBA-finals-assist-mapping-master/dubsG4asts.csv")
graph8 = readfile("/Users/xiaoy/Downloads/NBA-finals-assist-mapping-master/cavsG4asts.csv")

graph9 = readfile("/Users/xiaoy/Downloads/NBA-finals-assist-mapping-master/dubsG5asts.csv")
graph10 = readfile("/Users/xiaoy/Downloads/NBA-finals-assist-mapping-master/cavsG5asts.csv")

graph11 = readfile("/Users/xiaoy/Downloads/NBA-finals-assist-mapping-master/dubsG6asts.csv")
graph12 = readfile("/Users/xiaoy/Downloads/NBA-finals-assist-mapping-master/cavsG6asts.csv")

graph13 = readfile("/Users/xiaoy/Downloads/NBA-finals-assist-mapping-master/dubsG7asts.csv")
graph14 = readfile("/Users/xiaoy/Downloads/NBA-finals-assist-mapping-master/cavsG7asts.csv")

drawgraph(graph1, '#836654', '#836654', "GSW G1 assist")
drawgraph(graph2, '#e24b0a', '#830A0A', "CAVS G1 assist")

drawgraph(graph3, '#836654', '#836654', "GSW G2 assist")
drawgraph(graph4, '#e24b0a', '#830A0A', "CAVS G2 assist")

drawgraph(graph5, '#836654', '#836654', "GSW G3 assist")
drawgraph(graph6, '#e24b0a', '#830A0A', "CAVS G3 assist")

drawgraph(graph7, '#836654', '#836654', "GSW G4 assist")
drawgraph(graph8, '#e24b0a', '#830A0A', "CAVS G4 assist")

drawgraph(graph9, '#836654', '#836654', "GSW G5 assist")
drawgraph(graph10, '#e24b0a', '#830A0A', "CAVS G5 assist")

drawgraph(graph11, '#836654', '#836654', "GSW G6 assist")
drawgraph(graph12, '#e24b0a', '#830A0A', "CAVS G6 assist")

drawgraph(graph13, '#836654', '#836654', "GSW G7 assist")
drawgraph(graph14, '#e24b0a', '#830A0A', "CAVS G7 assist")
