import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time

def edgesFromGraph(graph):
    edges = []
    for node in graph:
        for edge in graph[node]:
            if sorted((node, edge)) not in edges:
                edges.append(sorted((node, edge)))
    return [tuple(edge) for edge in edges]

def verticesFromGraph(graph):
    return list(graph.keys())

def printCliqueDict(cliqueDict):
    for dimension in cliqueDict:
        #print(len(cliqueDict[dimension]))
        print("Dimension " + str(dimension) + " cliques: " + str(cliqueDict[dimension]))

def visualizeGraph(graph):
    G = nx.Graph()
    for node in graph:
        G.add_node(node)
        for edge in graph[node]:
            G.add_edge(node, edge)
    nx.draw(G, with_labels=True)
    plt.show()

def is_clique(b, graphMatrix, store):
    for i in range(1, b) :
        for j in range(i + 1, b):
            if (graphMatrix[store[i]][store[j]] == 0):
                return False     
    return True
 
def formatClique(n, store):
    out = []
    for i in range(1, n):
        out.append(store[i]-1)
    return out

def findCliques(i, l, sizeOfClique, nOfV, store, degrees, graphMatrix, out):
    for j in range(i + 1, nOfV - (sizeOfClique - l) + 1):
        if (degrees[j] >= sizeOfClique - 1):
            store[l] = j
            if (is_clique(l + 1, graphMatrix, store)):
                if (l < sizeOfClique):
                    findCliques(j, l + 1, sizeOfClique, nOfV, store, degrees, graphMatrix, out)
                else:
                    out.append(formatClique(l + 1, store))
    return out

def cliques(VG, EG):
    MAX = 100
    outDict = {}
    for k in range(1, len(VG)+1):
        store = [0] * MAX
        degrees = [0] * MAX
        graphMatrix = np.zeros((MAX, MAX))
        for i in range(len(EG)):
            graphMatrix[EG[i][0]][EG[i][1]] = 1
            graphMatrix[EG[i][1]][EG[i][0]] = 1
            degrees[EG[i][0]] += 1
            degrees[EG[i][1]] += 1
        cliq = findCliques(0, 1, k, len(VG), store, degrees, graphMatrix, [])
        if len(cliq) > 0:
            outDict[k] = cliq
        if len(cliq) == 0:
            break
    return outDict


def VR(S, epsilon):
    EG = []
    for i in range(len(S)):
        for j in range(i+1, len(S)):
            if np.linalg.norm(np.array(S[i]) - np.array(S[j])) <= epsilon:
                EG.append((i, j))

    G = nx.Graph()
    G.add_edges_from(EG)
    nx.draw(G, with_labels=True)
    plt.show()
    
    VG = [i+1 for i in range(len(S))]
    EG = [(x+1, y+1) for (x,y) in EG]
    return cliques(VG, EG)




### TESTS ###

#tetrahedra
graph = {
    0: [1, 2, 3],
    1: [0, 2, 3],
    2: [0, 1, 3],
    3: [0, 1, 2]
}
edges = edgesFromGraph(graph)
vertices = verticesFromGraph(graph)
VG = [i+1 for i in vertices]
EG = [(x+1, y+1) for (x,y) in edges]
tetrahedraCliques = cliques(VG, EG)
printCliqueDict(tetrahedraCliques)
visualizeGraph(graph)



#complete graph of 16 vertices
initTime = time.time()
graph2 = nx.complete_graph(16)
VG = [i+1 for i in list(graph2.nodes())]
EG = [(x+1, y+1) for (x,y) in list(graph2.edges())]
graph2Cliques = cliques(VG, EG)
printCliqueDict(graph2Cliques)
endTime = time.time()
execTime = round(endTime-initTime, 2)
print(f"The execution time is {execTime} seconds.")
visualizeGraph(graph2)



#complete graph of 19 vertices
initTime = time.time()
graph3 = nx.complete_graph(19)
VG = [i+1 for i in list(graph3.nodes())]
EG = [(x+1, y+1) for (x,y) in list(graph3.edges())]
graph3Cliques = cliques(VG, EG)
printCliqueDict(graph3Cliques)
endTime = time.time()
execTime = round(endTime-initTime, 2)
print(f"The execution time is {execTime} seconds.")
visualizeGraph(graph3)



#complete graph of 22 vertices
initTime = time.time()
graph4 = nx.complete_graph(22)
VG = [i+1 for i in list(graph4.nodes())]
EG = [(x+1, y+1) for (x,y) in list(graph4.edges())]
graph4Cliques = cliques(VG, EG)
printCliqueDict(graph4Cliques)
endTime = time.time()
execTime = round(endTime-initTime, 2)
print(f"The execution time is {execTime} seconds.")
visualizeGraph(graph4)


#vietoris-rips complex of 6 given points
S = [(0, 0), (1, 1), (2, 3), (-1, 2), (3, -1), (4, 2)]
epsilon = 3
sampleVR = VR(S, epsilon)
printCliqueDict(sampleVR)


#vietoris-rips complex of 10 random points
np.random.seed(3)
S1 = [(np.random.randint(0, 10), np.random.randint(0, 10)) for _ in range(10)]
vr1 = VR(S1, epsilon)
printCliqueDict(vr1)



#vietoris-rips complex of 20 random points
np.random.seed(10)
S2 = [(np.random.randint(0, 10), np.random.randint(0, 10)) for _ in range(20)]
vr2 = VR(S2, epsilon)
printCliqueDict(vr2)