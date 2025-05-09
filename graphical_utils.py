# -*- coding: utf-8 -*-
#import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

# Funciones auxiliares para graficar

def grafica_fromCompleteVects(x,y, activateGrid=True, showLinks=True):

    # turn into numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)

    # must have same shape except for dim
    if x.ndim == 1 and y.ndim == 1:
        # single trajectory: promote to a single-row 2D array
        x = x[np.newaxis, :]
        y = y[np.newaxis, :]
    elif x.ndim == 2 and y.ndim == 2:
        # already multiple trajectories: OK
        pass
    else:
        raise ValueError("x and y must both be either 1D (shape (n,)) or 2D (shape (m, n)) arrays")

    #https://numpy.org/doc/stable/reference/generated/numpy.insert.html
    #primero va el indice, y luego lo que quieres agregar. si es escalar, lo repite varias veces
    x = np.insert(x, 0,0, axis=1)
    y = np.insert(y, 0,0, axis=1)

    fig = plt.figure()

    for x_i, y_i in zip(x,y):
        s = plt.scatter(x_i,y_i)
        if showLinks: s = plt.plot(x_i,y_i)

    plt.scatter(0,0, s=123, c='k')
    plt.axis('square') #https://stackoverflow.com/questions/17990845/how-to-equalize-the-scales-of-x-axis-and-y-axis-in-matplotlib
    plt.grid(activateGrid)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title("Cartesian World Space Plot of SCARA Robot 2D")
    plt.show()

    pass

def grafica_polar_fromCompleteVects(r,theta):

    # turn into numpy arrays
    r = np.asarray(r)
    theta = np.asarray(theta)

    # must have same shape except for dim
    if r.ndim == 1 and theta.ndim == 1:
        # single trajectory: promote to a single-row 2D array
        r = r[np.newaxis, :]
        theta = theta[np.newaxis, :]
    elif r.ndim == 2 and theta.ndim == 2:
        # already multiple trajectories: OK
        pass
    else:
        raise ValueError("r and theta must both be either 1D (shape (n,)) or 2D (shape (m, n)) arrays")

    #https://numpy.org/doc/stable/reference/generated/numpy.insert.html
    #primero va el indice, y luego lo que quieres agregar. si es escalar, lo repite varias veces
    r = np.insert(r, 0,0, axis=1)
    theta = np.insert(theta, 0,0, axis=1)

    fig = plt.figure()

    for r_i, theta_i in zip(r,theta):
        s = plt.polar(theta_i, r_i)
        #s = plt.plot(theta_i, r_i)

    plt.scatter(0,0, s=123, c='k')
    plt.title("Polar World Space Plot of SCARA Robot 2D")
    plt.show()

    pass

####################################################################################################

import seaborn as sns
#import matplotlib.pyplot as plt
#import numpy as np

def convert_maze_toHeatmap(maze_asCharListOfLists, char_to_num=None, cmap='Set1', fSize=(10, 7), showDescriptiveLegend=True,
                                    showColorBar=False, color_of_blocked_state='ffe400'):

    if char_to_num is None:
        print("Using default character-to-number values")
        char_to_num = {
            ' ': 1, # free state in the c-space
            '~': 3, # unknown (unexplored) state in the c-space
            '#': np.nan, # blocked state in the c-space
            'H': 2, # blocked goal state in the c-space
            'Z': 4, # blocked start state in the c-space
            'S': 0, # start state in the c-space
            'G': 6, # goal state in the c-space
        }

        descriptions = {
            ' ': 'feasible state',
            '~': 'unknown state',
            '#': 'collision',
            'H': 'blocked goal',
            'Z': 'blocked start',
            'S': 'start state',
            'G': 'goal state',
        }

        print("\nChar: Value -> Description of the (c-space) cell")
        for char, value in char_to_num.items():
            print(f"'{char}': {value} -> {descriptions[char]}")
        print()

    # normalize color_of_blocked_state to a hex string
    blocked_color_as_hexStr = (
        color_of_blocked_state
        if color_of_blocked_state.startswith('#')
        else f'#{color_of_blocked_state}'
    )

    char_list = maze_asCharListOfLists

    # Convert the list of lists of characters to a 2D numpy array of numerical values
    num_array = np.array([[char_to_num[char] for char in sublist] for sublist in char_list])

    lista_wallCharacters = ["#"]
    wall_cells = []

    # Plot the heatmap
    plt.figure(figsize=fSize)
    hMap = sns.heatmap(num_array, annot=False, fmt='d', cmap=cmap, cbar=showColorBar,
                xticklabels=False, yticklabels=False)

    # Optional: Add the original characters as annotations
    for i in range(len(char_list)):
        for j in range(len(char_list[i])):
            plt.text(j + 0.5, i + 0.5, char_list[i][j],
                    ha='center', va='center', color='black')

            if char_list[i][j] in lista_wallCharacters: wall_cells.append((i,j))

            if char_list[i][j] == 'S': start_cell = (i, j)
            if char_list[i][j] == 'G': end_cell = (i, j)

    print("Start cell:", start_cell)
    print("End cell:", end_cell)

    # Overlay of the cells that are walls
    for (i, j) in wall_cells:
        hMap.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, facecolor=blocked_color_as_hexStr, edgecolor='gray'))

    if showDescriptiveLegend:
        import matplotlib.patches as mpatches  # for legend patches

        # get the QuadMesh (the actual ScalarMappable) instead of the Axes
        quadmesh = hMap.collections[0]
        norm = quadmesh.norm
        cmap = quadmesh.cmap

        # grab the Colorbar instance that seaborn already attached to the mappable
        cbar = quadmesh.colorbar

        # build legend handles for each cell-type
        handles = []
        for char, desc in descriptions.items():
            val = char_to_num[char]
            if np.isnan(val):
                # for blocked cells we know we painted them gray
                handles.append(mpatches.Patch(color=blocked_color_as_hexStr, label=desc))
            else:
                handles.append(
                    mpatches.Patch(color=cmap(norm(val)), label=desc)
                )


        # figure & axes bbox in figure coords
        fig = hMap.figure

        if showColorBar:
            # --- place legend next to the colorbar ---
            cbar = quadmesh.colorbar
            cbar_bbox = cbar.ax.get_position()  # in figure coords
            legend_x = cbar_bbox.x1 + 0.01       # small gap
            legend_y = cbar_bbox.y0 + cbar_bbox.height / 2

            cbar.ax.legend(
                handles=handles,
                title="Cell Types",
                loc='center left',
                bbox_to_anchor=(legend_x, legend_y),
                bbox_transform=fig.transFigure,
                frameon=False
            )

        else:
            # --- shrink heatmap to make room for legend ----
            hm_bbox = hMap.get_position()
            new_width = hm_bbox.width * 0.8
            hMap.set_position([hm_bbox.x0, hm_bbox.y0, new_width, hm_bbox.height])

            legend_x = hm_bbox.x0 + new_width + 0.02
            legend_y = hm_bbox.y0 + hm_bbox.height / 2

            # draw legend on the figure itself
            fig.legend(
                handles=handles,
                title="Cell Types",
                loc='center left',
                bbox_to_anchor=(legend_x, legend_y),
                bbox_transform=fig.transFigure,
                frameon=False
            )

    plt.show()

    return num_array

####################################################################################################

# Creamos clases para nodos y problemas simulados
class Node_Simulated():
  def __init__(self, state):
    self.state = state

class Problem_Simulated():
  def __init__(self, goal):
    self.goal = goal

import itertools

def display_pathCosts_ofResults(maze, visitedNodes, solutionNodes, valToPlot='cost',
                            tipoDeBusq=None, hFunc=None, cmap='plasma', addNumericVals=False, fSize=(10, 7)):
    """ Muestra los resultados de path cost luego de realizar la búsqueda en el maze.   """

    title_sufix = "" if tipoDeBusq is None else "Alg. de Búsq.: %s - " % (tipoDeBusq)

    grid_ofPathCosts = []
    solution_path_cells = []

    lista_wallCharacters = ["#"]
    wall_cells = []

    grid_ofHeuristicVals = []
    goal_state = solutionNodes[-1].state # visited_nodes[-1] and solutionNodes[-1] should result in the same state

    for i, row in enumerate(maze.grid):
        grid_ofPathCosts.append([np.nan for x in row])
        grid_ofHeuristicVals.append([np.nan for x in row])
        wall_cells.append([(i,j) for j, x in enumerate(row) if x in lista_wallCharacters])

    # https://stackoverflow.com/questions/952914/how-do-i-make-a-flat-list-out-of-a-list-of-lists
    wall_cells = list(itertools.chain.from_iterable(wall_cells))

    for node in visitedNodes:

        row,col = node.state
        path_cost_ofThisNode = node.path_cost
        grid_ofPathCosts[row][col] = path_cost_ofThisNode

        heuristic_value_ofThisNode = hFunc(Node_Simulated(node.state),
                                                    Problem_Simulated(goal_state)) if hFunc is not None else 0
        grid_ofHeuristicVals[row][col] = round(heuristic_value_ofThisNode, 1)
    for node in solutionNodes:
        row,col = node.state
        solution_path_cells.append( (row, col) )

    grid_ofPathCosts = np.array(grid_ofPathCosts)
    grid_ofHeuristicVals = np.array(grid_ofHeuristicVals)

    # Inicio del Ploteo
    plt.figure(figsize=fSize)

    if valToPlot=='cost':
        plt.title(title_sufix+"Heatmap de: Costo de Acciones (g)")
        gridMovil = grid_ofPathCosts

    elif valToPlot=='heuristic':
        plt.title(title_sufix+"Heatmap de: Heurística (h)")
        gridMovil = grid_ofHeuristicVals

    elif valToPlot=='evalFun':
        plt.title(title_sufix+"Heatmap de: Evaluation Function (f = g + h)")
        gridMovil = grid_ofPathCosts + grid_ofHeuristicVals

    else:
        raise ValueError("%s must be in ['cost', 'heuristic', 'evalFun']")

    # https://stackoverflow.com/questions/74415190/how-to-have-only-int-annotations-in-seaborn-heatmap
    hMap = sns.heatmap(gridMovil, annot=False, cmap=cmap, cbar=True,
                xticklabels=False, yticklabels=False)

    # Optional: Add the values as text
    if addNumericVals:
        for i in range(len(gridMovil)):
            for j in range(len(gridMovil[i])):
                plt.text(j + 0.5, i + 0.5, gridMovil[i][j],
                        ha='center', va='center', color='black', fontsize='xx-small')

    # Overlay of the cells that are walls
    for (i, j) in wall_cells:
        hMap.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, facecolor='gray', edgecolor='gray'))

    # Overlay of the cells that belong to the solution path found
    for (i, j) in solution_path_cells:
        hMap.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, lw=0.5, edgecolor='black'))

    # Overlay of the start and end cells:
    hMap.add_patch(plt.Rectangle(solution_path_cells[ 0][::-1], 1, 1, fill=False,
                                 lw=2, linestyle = ':', edgecolor='red'))
    hMap.add_patch(plt.Rectangle(solution_path_cells[-1][::-1], 1, 1, fill=False,
                                 lw=2, linestyle = '-', edgecolor='red'))

    plt.show()

    return grid_ofPathCosts, grid_ofHeuristicVals

if __name__ == "__main__":
    # Sample list of lists of characters
    char_list = [
        ['H', 'Z', ' ', ' '],
        ['S', '~', '#', ' '],
        ['G', ' ', '#', '#']
    ]

    # Create a mapping from characters to numerical values
    convert_maze_toHeatmap(char_list, char_to_num=None, showColorBar=False) # ejm para entender cómo funciona el ploteador