import networkx as nx
from numpy.core.numeric import cross
import osmnx as ox
from osmnx import distance as distance
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import csv
import random
import operator
import math
import pandas
import matplotlib.pyplot as plt
from collections import deque
from copy import deepcopy

## -- Set up the initial map area and save it as a networkx graph
ox.config(log_console=True, use_cache=True)
G = ox.graph_from_address('1276 Gilbreath Drive, Johnson City, Washington County, Tennessee, 37614, United States', dist=8000, network_type='drive')

### Use this code to display a plot of the graph if desired. Note: You need to import matplotlib.pyplot as plt
# fig, ax = ox.plot_graph(G, edge_linewidth=3, node_size=0, show=False, close=False)
# plt.show()

## -- Genetic Algorithm Parameters
GENERATIONS = 100
POPULATION_SIZE = 200
MUTATION_RATE = 0.1
DISPLAY_RATE = 20

## -- Set up Origin and Destination Points
origin_point = (36.3044549, -82.3632187) # Start at ETSU
destination_point= (36.3044549,	-82.3632187) # End at ETSU
origin = ox.get_nearest_node(G, origin_point)
destination = ox.get_nearest_node(G, destination_point)
origin_node = (origin, G.nodes[origin])
destination_node = (destination, G.nodes[destination])

## -- Set up initial lists
points = []                 # The list of osmnx nodes that can be used for map plotting
generations = []            # A list of populations, ultimately of size GENERATIONS
population = []             # The current population of size POPULATION_SIZE
chromosome = []             # Represented as a list of index values that correspond to the points list

def plot_path(lat, long, origin_point, destination_point, fitness):
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
        marker=go.scattermapbox.Marker(
            size=9
        ),
        text=[str(i) for i in range(1,len(long))],
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
        marker={'size': 12, 'color': 'green'},
        text=str(fitness)))

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
                          'zoom': 12})
    fig.show()

# Modified to take a crossover strategy and gensave the image instead of display it, and return the data dict
def plot_ga(crossover_strategy):
    generation_values = []
    best = []
    worst = []
    gen = 1
    for g in generations:
        best_route = g[0]
        worst_route = g[POPULATION_SIZE-1]
        best.append(best_route[1])
        worst.append(worst_route[1])
        generation_values.append(gen)
        gen = gen+1
    temp_data = {'Best':best,'Worst':worst }
    df = pandas.DataFrame(temp_data)
    plot = df.plot(title=f"Fitness Across Generations: {crossover_strategy} crossover", xlabel="Generatons", ylabel="Fitness")
    plot.figure.savefig(f"FitnessAcrossGenerations_{crossover_strategy}-crossover.png")
    plt.clf()

    return temp_data

# Takes a data dict indexed by strategy name containing data from algorithm runs and creates a line plot comparing the best runs from each, saving as a png
def plot_bests(data):
    best_data = {}
    for strategy, results in data.items():
        best_data[strategy] = results['Best']

    df = pandas.DataFrame(data)
    fig = px.line(df, x="Generations", y=data.keys(), title="Fitness Across Generations: Best From Each Strategy")
    fig.write_image("Fitness Across Generations: Best From Each Strategy")    

# Modified to take in actual points, not ids, no need to perform look up twice
def haversine(point1, point2):
    """
    Returns the Great Circle Distance between point 1 and point 2 in miles
    """
    return ox.distance.great_circle_vec(point1['y'], point1['x'], point2['y'], point2['x'], 3963.1906)

# Returns a list containing chromosome,fitness ready to be inserted into a population
def calculate_fitness(chromosome):
    """
    Fitness is the total route cost using the haversine distance.
    The GA should attempt to minimize the fitness; minimal fitness => best fitness
    """
    fitness = 0.0
    
    for point in range(len(chromosome) - 1):
        fitness += haversine(chromosome[point][1], chromosome[point + 1][1])


    return [chromosome,fitness]


## initialize population
def initialize_population():
    """
    Initialize the population by creating POPULATION_SIZE chromosomes.
    Each chromosome represents the index of the point in the points list.
    Sorts the population by fitness and adds it to the generations list.
    """
    my_population = []

    # Loop through creating chromosomes until we fill the population
    for chromosome in range(0, POPULATION_SIZE):
        # Shuffle the list of points and calculate the fitness of the path which returns the [chromosme,fitness] ready to be added to the population
        my_population.append(calculate_fitness(random.sample(points, len(points))))     

    # Sort the population by fitness
    my_population.sort(key=lambda x: x[1])

    generations.append(my_population)

# Takes the index to the generation to repopulate from, and a crossover strategy (accepts: uniform, singlepoint, multipoint)
def repopulate(gen, crossover_strategy, random_selection=False):
    """
    Creates a new generation by repopulation based on the previous generation.
    Calls selection, crossover, and mutate to create a child chromosome. Calculates fitness
    and continues until the population is full. Sorts the population by fitness
    and adds it to the generations list.
    """
    ## Ensure you keep the best of the best from the previous generation
    retain = math.ceil(POPULATION_SIZE*0.025)
    new_population = generations[gen-1][:retain]

    ## Conduct selection, reproduction, and mutation operations to fill the rest of the population
    while len(new_population) < POPULATION_SIZE:
        # Select the two parents from the growing population
        parent1, parent2 = selection(gen, random_selection)
        # Generate the child according to the designated crossover_strategy
        child = crossover(parent1, parent2, crossover_strategy)
        # Generate a random number, if it falls beneath the mutation_rate, perform a point swap mutation on the child
        if (random.random() < MUTATION_RATE):
            child = mutate(child[0])
            
        new_population.append(child)

    # Sort the population by fitness
    new_population.sort(key=lambda x: x[1])

    generations.append(new_population)

# Adopted and modified from Genetic Search Algorithm lab
# Set rand to True to divert typical functionality and choose parents completely at random
def selection(gen, rand=False):
    '''
    Selects parents from the given population, assuming that the population is
    sorted from best to worst fitness.

    Parameters
    ----------
    population : list of lists
        Each item in the population is in the form [chromosome,fitness]

    Returns
    -------
    parent1 : list of int
        The chromosome chosen as parent1
    parent2 : list of int
        The chromosome chosen as parent2

    '''
    # Set the elitism factor and calculate the max index
    if rand == False:
        factor = 0.5	# Select from top 50%
        high = math.ceil(POPULATION_SIZE*factor)
    else:
        high = POPULATION_SIZE - 1

    # Choose parents randomly
    parent1 = generations[gen-1][random.randint(0,high)][0]
    parent2 = generations[gen-1][random.randint(0,high)][0]

    # If the same parent is chosen, pick another
    # we can get stuck here if we converge early, if we pick the same parent ten times in a row, just bail out
    count = 0
    while str(parent1) == str(parent2):
        parent2 = generations[gen-1][random.randint(0,high)][0]
        count += 1
        if count == 10:
            break

    return parent1, parent2

# Adopted and modified from Genetic Search Algorithm lab
# Set crossover_strategy to "singlepoint"/"multipoint" to divert from typical behavior and instead perform a singlepoint/multipoint reproduction strategy
def crossover(parent1, parent2, crossover_strategy="uniform"):
    '''
    Parameters
    ----------
    parent1 : list of int
        A chromosome that lists the steps to take
    parent2 : list of int
        A chromosome that lists the steps to take

    Returns
    -------
    list in the form [chromosome,fitness]
        The child chromosome and its fitness value

    '''
    # Initialization
    child = []
    chromosome_size = len(parent1)
    if crossover_strategy == "singlepoint":
        # Randomly choose a split point
        split_point = chromosome_size - random.randint(0, chromosome_size)
        child = parent1[:split_point] + parent2[split_point:]
    elif crossover_strategy == "multipoint":
        points = []
        while len(points) < 2: 
            split_point = chromosome_size - random.randint(0, chromosome_size) 
            if split_point not in points:
                points.append(split_point)
        points.sort()
        child = parent1[:points[0]] + parent2[points[0]:points[1]] + parent1[points[1]:]
    else:
        # Step through each item in the chromosome and randomly choose which
        #  parent's genetic material to select
        for i in range(0, chromosome_size):
            bit = None
            if random.randint(0,1) == 0:
                bit = parent1[i]
            else:
                bit = parent2[i]
            child.append(bit)

    return calculate_fitness(child)


def mutate(chromosome):
    """
    Strategy: swap two pairs of points. Return the chromosome after mutation.
    """
    # Copy the child
    mutant_child = deepcopy(chromosome)
    # Select two random points
    point1, point2 = random.sample(range(len(mutant_child)), 2)
    #Swap the points
    mutant_child[point1], mutant_child[point2] = mutant_child[point2], mutant_child[point1]
    
    return calculate_fitness(mutant_child)

# Modified to rake a crossover strategy and random_selection flag (defaulted to False)
def run_ga(crossover_strategy, random_selection=False):
    """
    Initialize and repopulate until you have reached the maximum generations
    """
    initialize_population()

    for gen in range(GENERATIONS-1):      #Note, you already ran generation 1
        repopulate(gen+1, crossover_strategy, random_selection)
        if gen % DISPLAY_RATE == 0:
            print("Best Geneartion:") # Print the generation, and the best (lowest) fitness score in the population for that generation
            print(generations[gen])
            print("Fitness Score")
            print(generations[gen][0][1])

def show_route(generation_number):
    """
    Gets the latitude and longitude points for the best route in generation_number
    """
    the_route = generations[generation_number][0][0]
    the_fitness = generations[generation_number][0][1]

    startend = [g for g in G.nodes(True) if g[0] == origin_node[0]][0]

    route = [startend[0]]
    lat = [startend[1]["y"]]
    long = [startend[1]["x"]]

    for p in the_route:
        node = [g for g in G.nodes(True) if g[0] == points[p][0]][0]
        route.append(node[0])
        lat.append(points[p][1]["y"])
        long.append(points[p][1]["x"])

    route.append(startend[0])
    lat.append(startend[1]["y"])
    long.append(startend[1]["x"])
    plot_path(lat,long,origin_node,destination_node,the_fitness)
    print("The fitness for generation",generation_number,"was",the_fitness)


def main():
    """
    Reads the csv file and then runs the genetic algorithm and displays results.
    """
    with open('addresses_geocoded.csv') as file:

        csvFile = csv.reader(file)

        for xy in csvFile:
            point_coordinates = (float(xy[0]),float(xy[1]))
            point = ox.get_nearest_node(G, point_coordinates)
            point_node = (point, G.nodes[point])
            if point_node not in points:
                points.append(point_node)

    print("There were",len(points),"unique points found.")

    strategies = ['uniform', 'singlepoint', 'multipoint']
    data = {}

    for current_strategy in strategies:
        run_ga(crossover_strategy=current_strategy, random_selection=False)
        #show_route(0)
        #show_route(math.floor(GENERATIONS/2))
        #show_route(GENERATIONS-1)

        temp_data = plot_ga(current_strategy)
        data[f"{current_strategy}-Best"] = temp_data['Best']
        # reset
        generations.clear()

    df = pandas.DataFrame(data)
    plot = df.plot(title="Comparing Best Runs Accross Crossover Strategies", xlabel="Generatons", ylabel="Fitness")
    plot.figure.savefig("Comparing-Best-Runs-Accross-Crossover-Strategies.png")

main()
