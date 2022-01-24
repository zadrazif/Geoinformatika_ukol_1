import shapefile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

# read input file
sf = shapefile.Reader(r'obce_tabor.shp')

# grab the shapefile's field names (omit the first pseudo field) 
fields = [x[0] for x in sf.fields][1:]
records = sf.records()
shps = [s.points for s in sf.shapes()]

# write the records into a dataframe 
shp_dataframe = pd.DataFrame(columns=fields, data=records)

# add the coordinate data to a column called "coords" 
df = shp_dataframe.assign(coords=shps)
coords = df['coords'].values.tolist()

# Function for the Best Insertion heuristic
def BI(coords):
    # initialize unprocessed nodes, processed nodes and sum of weights (W)
    u_nodes = coords.copy()
    p_nodes = []
    W = 0

    # choose three random points to create a circle
    for u in range(3):
        random.shuffle(u_nodes)
        u_i = np.array(u_nodes.pop())
        p_nodes.append(u_i)

    # save starting position
    first = p_nodes[0]

    # calculate distance between first three nodes
    for i in range(len(p_nodes)):
        u1 = p_nodes[i]
        u2 = (p_nodes + [first])[i+1]
        dist = np.linalg.norm(u1 - u2)
        W += dist

    # go through unprocessed nodes until all nodes are processed
    while len(u_nodes) != 0:
        # catch out of index exception
        hamilton = p_nodes + [first]

        # pick a node which is closest to the hamiltonian circle
        abs_min = float("inf")
        index = -1
        counter = 0
        for i in u_nodes:
            dist = 0
            for j in range(len(p_nodes)):
                d_w = np.linalg.norm(hamilton[j] - i) + np.linalg.norm(i - hamilton[j+1]) - np.linalg.norm(hamilton[j] - hamilton[j+1])
                dist += d_w
            if d_w < abs_min:
                abs_min = dist
                index = counter
            counter += 1

        u = np.array(u_nodes.pop(index))
        
        # initialize list of delta w distances
        d_ws = []

        # calculating delta w for every edge
        for i in range(len(p_nodes)):
            d_w = np.linalg.norm(hamilton[i] - u) + np.linalg.norm(u - hamilton[i+1]) - np.linalg.norm(hamilton[i] - hamilton[i+1])
            d_ws.append(d_w)

        # find minimal delta w and index of this node
        min_w = min(d_ws)
        min_i = d_ws.index(min_w)

        # insert node to hamilton's circle
        p_nodes.insert(min_i+1, u)
        W += min_w

    # add starting position to the end of list (creating Hamiltonian circle)
    p_nodes.append(first)

    # print overall weight of the path
    print("W of BI [km]: ", W/1000)
    return W, p_nodes 

if __name__ == '__main__':
    # compute weights and nodes
    weight_BI, nodes_BI = BI(coords)
    
    labels_BI = map(str, list(range(1, len(nodes_BI))))
    x_nodes_BI = [e[0][0] for e in nodes_BI]
    y_nodes_BI = [e[0][1] for e in nodes_BI]

    # create plot of BI
    plt.figure()
    plt.scatter(x_nodes_BI, y_nodes_BI)
    plt.plot(x_nodes_BI, y_nodes_BI)
    for x, y, l in zip(list(x_nodes_BI), list(y_nodes_BI), labels_BI):
        plt.text(x, y, l)
    plt.show()