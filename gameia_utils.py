####################################################################################################
# MAZE GRID UTILS
####################################################################################################

class Maze:
    def __init__(self, grid):
        """ Construye el maze a partir del grid ingresado
            grid: debe ser una matriz (lista de listas), ejemplo [['#','S',' '],['#',' ','G']]  """
        self.grid = grid.copy()
        self.numRows = len(grid)
        self.numCols = len(grid[0])
        for i in range(self.numRows):
            for j in range(self.numCols):
                if len(grid[i]) != self.numCols:
                    raise "Grid no es Rectangular"
                if grid[i][j] == 'S':
                    self.startCell = (i,j)
                if grid[i][j] == 'G':
                    self.exitCell= (i,j)
        if self.exitCell == None:
            raise "No hay celda de Inicio"
        if self.startCell == None:
            raise "No hay celda de salida"

    def isPassable(self, row, col):
        """ Retorna true si la celda (row,col) es pasable  (' ' o '~' o '=') """
        return self.isWater(row, col) or self.isSand(row,col) or self.isClear(row, col)

    def isWater(self,row,col):
      return self.grid[row][col] == '~'

    def isSand(self,row,col):
      return self.grid[row][col] == '='

    def isClear(self, row, col):
        """ Retorna true si la celda (row,col) esta vacia  (' ') """
        return self.grid[row][col] == ' '

    def isBlocked(self, row,col):
        """ Retorna true si la celda (row,col) tiene obstaculo ('#') """
        return self.grid[row][col] == '#'

    def getNumRows(self):
        """ Retorna el numero de filas en el maze """
        return self.numRows

    def getNumCols(self):
        """ Retorna el numero de columnas en el maze """
        return self.numCols

    def getStartCell(self):
        """ Retorna la posicion (row,col) de la celda de inicio """
        return self.startCell

    def getExitCell(self):
        """ Retorna la posicion (row,col) de la celda de salida """
        return self.exitCell

    def __getAsciiString(self):
        """ Retorna el string de vizualizacion del maze """
        lines = []
        headerLine = ' ' + ('-' * (self.numCols)) + ' '
        lines.append(headerLine)
        for row in self.grid:
            rowLine = '|' + ''.join(row) + '|'
            lines.append(rowLine)
        lines.append(headerLine)
        return '\n'.join(lines)

    def __str__(self):
        return self.__getAsciiString()
    

class SearchProblem(object):
    def __init__(self, initial, goal=None):
        """Este constructor especifica el estado inicial y posiblemente el estado(s) objetivo(s),
        La subclase puede añadir mas argumentos."""
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        """Retorna las acciones que pueden ser ejecutadas en el estado dado.
        El resultado es tipicamente una lista."""
        raise NotImplementedError

    def result(self, state, action):
        """Retorna el estado que resulta de ejecutar la accion dada en el estado state.
        La accion debe ser alguna de self.actions(state)."""
        raise NotImplementedError

    def goal_test(self, state):
        """Retorna True si el estado pasado satisface el objetivo."""
        raise NotImplementedError

    def path_cost(self, c, state1, action, state2):
        """Retorna el costo del camino de state2 viniendo de state1 con
        la accion action, asumiendo un costo c para llegar hasta state1.
        El metodo por defecto cuesta 1 para cada paso en el camino."""
        return c + 1
    

class MazeSearchProblem(SearchProblem):
    def __init__(self, maze):
        """El constructor recibe el maze"""
        self.maze = maze
        self.initial = maze.getStartCell()
        self.goal = maze.getExitCell()
        self.numNodesExpanded = 0
        self.expandedNodeSet = {}

    def isValidState(self,state):
        """ Retorna true si el estado dado corresponde a una celda no bloqueada valida """
        raise NotImplementedError

    def actions(self, state):
        """Retorna las acciones legales desde la celda actual """
        raise NotImplementedError

    def result(self, state, action):
        """Retorna el estado que resulta de ejecutar la accion dada desde la celda actual.
        La accion debe ser alguna de self.actions(state)"""
        raise NotImplementedError

    def goal_test(self, state):
        """Retorna True si state es self.goal"""
        return (self.goal == state)

    def path_cost(self, c, state1, action, state2):
        raise NotImplementedError
    
def displayResults(maze, visitedNodes, solutionNodes):
    """ Muestra los resultados de busqueda en el maze.   """
    grid_copy = []
    for row in maze.grid:
        grid_copy.append([x for x in row])
    for node in visitedNodes:
        row,col = node.state
        ch = maze.grid[row][col]
        if ch != 'S' and ch != 'G': grid_copy[row][col] = 'o'
    for node in solutionNodes:
        row,col = node.state
        ch = maze.grid[row][col]
        if ch != 'S' and ch != 'G': grid_copy[row][col] = 'x'
    maze_copy = Maze(grid_copy)
    print (maze_copy)
    print ("x - celdas en la solucion")
    print ("o - celdas visitadas durante la búsqueda")
    print ("-------------------------------")

####################################################################################################
# SEARCH UTILS
####################################################################################################

from collections import deque

class FIFOQueue(deque):
    """Una cola First-In-First-Out"""
    def pop(self):
        return self.popleft()
    
class Node:
    def __init__(self, state, parent=None, action=None, path_cost=0):
        "Crea un nodo de arbol de busqueda, derivado del nodo parent y accion action"
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def expand(self, problem):
        "Devuelve los nodos alcanzables en un paso a partir de este nodo."
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        next = problem.result(self.state, action)
        return Node(next, self, action,
                    problem.path_cost(self.path_cost, self.state, action, next))

    def solution(self):
        "Retorna la secuencia de acciones para ir de la raiz a este nodo."
        return [node.action for node in self.path()[1:]]

    def path(self):
        "Retorna una lista de nodos formando un camino de la raiz a este nodo."
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    def __lt__(self, node):
        return self.state < node.state

    def __eq__(self, other):
        "Este metodo se ejecuta cuando se compara nodos. Devuelve True cuando los estados son iguales"
        return isinstance(other, Node) and self.state == other.state

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __hash__(self):
        return hash(self.state)

def graph_search(problem, frontier):
    frontier.append(Node(problem.initial))
    explored = set()     # memoria de estados visitados
    visited_nodes = []   # almacena nodos visitados durante la busqueda
    while frontier:
        node = frontier.pop()
        visited_nodes.append(node)
        if problem.goal_test(node.state):
            return node, visited_nodes
        explored.add(node.state)

        frontier.extend(child for child in node.expand(problem)
                        if child.state not in explored and
                        child not in frontier)
    return None

import heapq
import itertools
class FrontierPQ:
    "Una Frontera ordenada por una funcion de costo (Priority Queue)"

    def __init__(self, initial, costfn=lambda node: node.path_cost):
        self.heap = []
        self.states = {}
        self.costfn = costfn
        self.counter = itertools.count()
        self.add(initial)

    def add(self, node):
        cost = self.costfn(node)
        count = next(self.counter)
        heapq.heappush(self.heap, (cost, count, node))
        self.states[node.state] = (cost, node)

    def pop(self):
        while self.heap:
            cost, _, node = heapq.heappop(self.heap)
            if self.states.get(node.state, (None, None))[1] == node:
                del self.states[node.state]
                return node
        raise KeyError('pop from empty frontier')

    def replace(self, node):
        "Correct implementation of replace using heapify"
        cost = self.costfn(node)
        if node.state in self.states:
            old_cost, _ = self.states[node.state]
            if cost < old_cost:
                self.states[node.state] = (cost, node)
                # Reconstruct heap correctly:
                self.heap = [(c, count, n) if n.state != node.state else (cost, count, node)
                             for c, count, n in self.heap]
                heapq.heapify(self.heap)

    def __contains__(self, state):
        return state in self.states

    def __len__(self):
        return len(self.states)

def best_first_graph_search(problem, f):
    """Busca el objetivo expandiendo el nodo de la frontera con el menor valor de la funcion f. Memoriza estados visitados
    Antes de llamar a este algoritmo hay que especificar La funcion f(node). Si f es node.depth tenemos Busqueda en Amplitud;
    si f es node.path_cost tenemos Busqueda  de Costo Uniforme. Si f es una heurística tenemos Busqueda Voraz;
    Si f es node.path_cost + heuristica(node) tenemos A* """

    frontier = FrontierPQ( Node(problem.initial), f )  # frontera tipo cola de prioridad ordenada por f
    explored = set()     # memoria de estados visitados
    visited_nodes = []   # almacena nodos visitados durante la busqueda
    while frontier:
        node = frontier.pop()
        visited_nodes.append(node)
        if problem.goal_test(node.state):
            return node, visited_nodes
        explored.add(node.state)
        for action in problem.actions(node.state):
            child = node.child_node(problem, action)
            if child.state not in explored and child.state not in frontier:
                frontier.add(child)
            elif child.state in frontier:
                incumbent = frontier.states[child.state][1]  # <-- extract the node correctly from the tuple
                if f(child) < f(incumbent):
                    frontier.replace(child)

def astar_search(problem, heuristic):
    f = lambda node: node.path_cost + heuristic(node, problem)
    return best_first_graph_search(problem, f)

def nullheuristic(node, problem):
    return 0

####################################################################################################
# HEURISTIC UTILS
####################################################################################################

import numpy as np

def straightline_dist(node, problem):
    
    # TODO: Complete here your code
    
    return 0