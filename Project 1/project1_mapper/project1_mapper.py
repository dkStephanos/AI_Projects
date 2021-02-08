import networkx as nx
import osmnx as ox
from osmnx import distance as distance
import plotly.graph_objects as go
import collections
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import time

ALL_RESULTS = ""

def method_timing(func):
    def wrapper(*arg):
        global ALL_RESULTS
        t1 = time.time()
        res = func(*arg)
        t2 = time.time()
        output_string = '\n%s took %0.3f ms' % (func, (t2-t1)*1000.0)
        print(output_string)
        ALL_RESULTS += output_string
        return [res,(t2-t1)*1000.0]
    return wrapper

ox.config(log_console=True, use_cache=True)

G = ox.graph_from_address('1276 Gilbreath Drive, Johnson City, TN, USA', dist=4000, network_type='drive')

### Use this code to display a plot of the graph if desired. Note: You need to import matplotlib.pyplot as plt
fig, ax = ox.plot_graph(G, edge_linewidth=3, node_size=0, show=False, close=False)
plt.show()


def node_list_to_path(gr, node_list):
    """
    SOURCE: Modified from Priyam, Apurv (2020). https://towardsdatascience.com/find-and-plot-your-optimal-path-using-plotly-and-networkx-in-python-17e75387b873

    Given a list of nodes, return a list of lines that together
    follow the path
    defined by the list of nodes.
    Parameters
    ----------
    gr : networkx multidigraph
    node_list : list
        the route as a list of nodes
    Returns
    -------
    lines : list of lines given as pairs ( (x_start, y_start),
    (x_stop, y_stop) )
    """
    edge_nodes = list(zip(node_list[:-1], node_list[1:]))

    lines = []
    for u, v in edge_nodes:
        # if there are parallel edges, select the shortest in length
        if gr.get_edge_data(u, v) is not None:
            data = min(gr.get_edge_data(u, v).values(),
                       key=lambda x: x['length'])

            # if it has a geometry attribute
            if 'geometry' in data:
                # add them to the list of lines to plot
                xs, ys = data['geometry'].xy
                lines.append(list(zip(xs, ys)))
            else:
                # if it doesn't have a geometry attribute,
                # then the edge is a straight line from node to node
                x1 = gr.nodes[u]['x']
                y1 = gr.nodes[u]['y']
                x2 = gr.nodes[v]['x']
                y2 = gr.nodes[v]['y']
                line = [(x1, y1), (x2, y2)]
                lines.append(line)

    return lines

def path_to_file(graph, route, filepath):
    route_map = ox.plot_route_folium(graph, route)
    route_map.save(filepath)


def plot_path(lat, long, origin_point, destination_point):
    """
    SOURCE: Modified from Priyam, Apurv (2020). https://towardsdatascience.com/find-and-plot-your-optimal-path-using-plotly-and-networkx-in-python-17e75387b873

    Given a list of latitudes and longitudes, origin
    and destination point, plots a path on a map

    Parameters
    ----------
    lat, long: list of latitudes and longitudes
    origin_point, destination_point: co-ordinates of origin
    and destination
    Returns
    -------
    Nothing. Only shows the map.
    """
    origin = (origin_point[1]["y"], origin_point[1]["x"])
    destination = (destination_point[1]["y"], destination_point[1]["x"])
    # adding the lines joining the nodes
    fig = go.Figure(go.Scattermapbox(
        name="Path",
        mode="lines",
        lon=long,
        lat=lat,
        marker={'size': 10},
        line=dict(width=4.5, color='blue')))
    # adding source marker
    fig.add_trace(go.Scattermapbox(
        name="Source",
        mode="markers",
        lon=[origin[1]],
        lat=[origin[0]],
        marker={'size': 12, 'color': "red"}))

    # adding destination marker
    fig.add_trace(go.Scattermapbox(
        name="Destination",
        mode="markers",
        lon=[destination[1]],
        lat=[destination[0]],
        marker={'size': 12, 'color': 'green'}))

    # getting center for plots:
    lat_center = np.mean(lat)
    long_center = np.mean(long)
    # defining the layout using mapbox_style
    fig.update_layout(mapbox_style="stamen-terrain",
                      mapbox_center_lat=30, mapbox_center_lon=-80)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0},
                      mapbox={
                          'center': {'lat': lat_center,
                                     'lon': long_center},
                          'zoom': 13})
    fig.show()


def backtrack(graph, origin, end_node, explored):
    '''
    Accepts the graph, the origin and the node reached. Also accepts the list of
    explored/visited locations.

    It should work backwards to the origin from node, using explored, so you
    know exactly which path you took to locate the destination.

    You should return a list of osmids in the order of the path.
    It may also be helpful to track the latitudes and longitudes of each item on
    the path so you can easily move it to the plot function, along with the
    path cost (i.e., distance).
    '''
    print("Backtracking")

    lat = [origin[1]['y']]
    long = [origin[1]['x']]
    total_cost = 0
    path = [end_node[0]]
    node = explored[-1]

    while True:
        path.append(node[0])
        lat.append(graph.nodes[node[0]]['y'])
        long.append(graph.nodes[node[0]]['x'])
        if node[1] == origin[0]:
            break
        if node[1] is not None:
            total_cost += graph.get_edge_data(node[1], node[0])[0]['length']
        for explored_node in explored:
            if explored_node[0] == node[1]:
                node = explored_node
                break


    path.append(origin[0])
    lat.append(origin[1]['y'])
    long.append(origin[1]['x'])
    path.reverse()
    lat.reverse()
    long.reverse()

    return [path, lat, long, total_cost]

@method_timing
def depth_first_search(graph, origin, destination):
    '''
    Accepts the graph and the origin and destination points
    Returns the result of backtracking through the explored list when the
     destination is found.
    '''
    print("Depth First Search")

    frontier = deque()
    explored = []
    searching = True

    frontier.append((origin[0], None))

    while searching:
        if len(frontier) == 0:
            return -1

        frontier_node = frontier.pop()
        current_node = frontier_node[0]
        previous_node = frontier_node[1]

        if previous_node is not None:
            explored.append((current_node, previous_node))
        for neighbor in graph.neighbors(current_node):
            if (neighbor, current_node) not in frontier and (neighbor, current_node) not in explored:
                if neighbor == destination[0]:
                    searching = False
                    break    #Success
                frontier.append((neighbor, current_node))

    return backtrack(graph, origin, destination, explored)

@method_timing
def breadth_first_search(graph, origin, destination):
    '''
    Accepts the graph and the origin and destination points
    Returns the result of backtracking through the explored list when the
     destination is found.
    '''
    print("Breadth First Search")

    frontier = deque()
    explored = []
    searching = True

    frontier.appendleft((origin[0], None))

    while searching:
        if len(frontier) == 0:
            return -1

        frontier_node = frontier.pop()
        current_node = frontier_node[0]
        previous_node = frontier_node[1]

        if previous_node is not None:
            explored.append((current_node, previous_node))
        for neighbor in graph.neighbors(current_node):
            if (neighbor, current_node) not in frontier and (neighbor, current_node) not in explored:
                if neighbor == destination[0]:
                    searching = False
                    break    #Success
                frontier.appendleft((neighbor, current_node))

    return backtrack(graph, origin, destination, explored)

@method_timing
def uninformed_search(graph, origin, destination):
        '''
        Accepts the graph and the origin and destination points
        Returns the result of backtracking through the explored list when the
         destination is found.
        '''
        print("My Uninformed Search Algorithm")

        frontier = deque()
        explored = []
        searching = True

        frontier.append((origin[0], None))

        while searching:
            if len(frontier) == 0:
                return -1

            frontier_node = frontier.pop()
            current_node = frontier_node[0]
            previous_node = frontier_node[1]

            if previous_node is not None:
                explored.append((current_node, previous_node))
            neighbors = list(graph.neighbors(current_node))
            random.shuffle(neighbors)
            for neighbor in neighbors:
                if (neighbor, current_node) not in frontier and (neighbor, current_node) not in explored:
                    if neighbor == destination[0]:
                        searching = False
                        break    #Success
                    frontier.append((neighbor, current_node))

        return backtrack(graph, origin, destination, explored)

## -- Set up Destination Point
destination_points = [
    (36.359595, -82.398868),    # Walmart on West Market
    (36.342513, -82.373483),    # Target on North Roan
    (36.320831, -82.277667),    # Tweetsie Trail entrance
    (36.316574, -82.352577),    # Frieberg's German Restuarant
    (36.301605, -82.337822),    # Food City on South Roan
    (36.347904, -82.400772),    # Best Buy on Peoples Street
]

origin_point = (36.30321114344463, -82.36710826765649) # Gilbreath Hall
origin = ox.get_nearest_node(G, origin_point)
origin_node = (origin, G.nodes[origin])
map = 0
for destination_point in destination_points:
    map += 1
    ALL_RESULTS += f"\n\nResults for destination point: ({destination_point[0]},{destination_point[1]})\n--------------------------\n\n"
    destination = ox.get_nearest_node(G, destination_point)
    destination_node = (destination, G.nodes[destination])
    bfs_distance = 0
    dfs_distance = 0
    lat = []
    long = []
    
    bfs_route, lat, long, bfs_distance = breadth_first_search(G, origin_node, destination_node)[0]
    route_path = node_list_to_path(G, bfs_route)
    plot_path(lat, long, origin_node, destination_node)
    path_to_file(G, bfs_route, f'data/graph-map{map}-bfs.html')

    dfs_route, lat, long, dfs_distance = depth_first_search(G, origin_node, destination_node)[0]
    route_path = node_list_to_path(G, dfs_route)
    plot_path(lat, long, origin_node, destination_node)
    path_to_file(G, dfs_route, f'data/graph-map{map}-dfs.html')
    
    uninformed_route, lat, long, uninformed_distance = uninformed_search(G, origin_node, destination_node)[0]
    route_path = node_list_to_path(G, uninformed_route)
    plot_path(lat, long, origin_node, destination_node)
    path_to_file(G, uninformed_route, f'data/graph-map{map}-uninformed.html')
    
    ALL_RESULTS += "\nTotal Route Distance (BFS):" + str(bfs_distance)
    ALL_RESULTS += "\nTotal Route Distance (DFS):" + str(dfs_distance)
    ALL_RESULTS += "\nTotal Route Distance (Random-Depth):" + str(uninformed_distance)

    text_file = open("data/project1_mapper_results.txt", "w")
    text_file.write(ALL_RESULTS)
    text_file.close()