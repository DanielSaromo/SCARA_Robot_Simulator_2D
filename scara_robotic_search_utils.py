from gameia_utils import MazeSearchProblem, FIFOQueue, graph_search, astar_search, nullheuristic
from graphical_utils import display_pathCosts_ofResults
from scara_robotic_maze_utils import SCARA_Robotic_Maze
import numpy as np
from gameia_utils import straightline_dist, displayResults

from typing import Tuple

class SCARA_Robotic_MazeSearchProblem(MazeSearchProblem):
    def __init__(self, maze, cost_type=None):
        # 1) Inicializamos la parte de MazeSearchProblem
        super().__init__(maze)
        self.maze = maze
        self.cost_type = 'total_change_in_theta_vals' if cost_type is None else cost_type

        # 2) Verificamos que el initial discreto sea válido; si no, elegimos el primero factible
        if not self.isValidState(self.initial):
            valid = next((s for s in maze.startCells if self.isValidState(s)), None)
            if valid is None:
                raise ValueError("No hay celdas iniciales factibles en startCells")
            self.initial = valid

        # 3) Encontramos UNA sola goal-cell discreta que sea alcanzable desde self.initial
        from collections import deque
        frontier = deque([self.initial])
        visited  = {self.initial}
        reachable_goal = None
        while frontier:
            s = frontier.popleft()
            if s in maze.exitCells:
                reachable_goal = s
                break
            for a in self.actions(s):
                ns = self.result(s, a)
                if self.isValidState(ns) and ns not in visited:
                    visited.add(ns)
                    frontier.append(ns)

        if reachable_goal is None:
            raise ValueError("No hay ninguna exitCell alcanzable desde initial")
        self.goal = [reachable_goal]   # lista con una sola meta

        # 4) Resto de la inicialización
        self.numNodesExpanded = 0
        self.expandedNodeSet  = {}


    def isValidState(self, state) -> bool:
        row, col = state

        # TODO: Implement this function. You can use self.maze.isBlocked(...)

        return True

    def actions(self, state) -> list:
        """Return all valid moves (4-neighbour + 4 diagonals)."""

        # TODO: Implement this function

        moves = ['N']

        return moves
    
    def result(self, state, action) -> Tuple[int, int]:
        r, c = state

        # TODO: Implement this function

        return (r, c)

    def goal_test(self,state):
        return state in self.goal
    
    def path_cost(self, c, state1, action, state2):
        """
        Retorna el costo acumulado hasta state2 partiendo de state1, usando la acción dada.
        Parámetros:
          c          - costo acumulado hasta state1
          state1     - tupla (r1, c1) en el espacio de configuración antes de la acción
          action     - 'N','S','E','W', 'NE', 'NW', 'SE', 'SW'
          state2     - tupla (r2, c2) en el espacio de configuración después de la acción
          cost_type  - tipo de costo a usar:
                        * 'total_change_in_theta_vals'
                        * 'dist_to_goal_in_ws'

        Retorna el costo acumulado hasta state2 partiendo de state1 usando la acción dada.
        Para cost_type='dist_to_goal_in_ws' acumula la distancia euclídea recorrida por el EE.
        """
        # 1) Extraer índices antes y después de ejecutar la acción
        r1, c1 = state1
        r2, c2 = state2

        # 2) Penalizar si la celda no factible
        cell_char = self.maze.grid[r2][c2]
        if cell_char in ('H', 'Z'):
            step_cost = 10
        else:
            # 3) Calcular step_cost según cost_type
            ct = self.cost_type

            if ct == 'total_change_in_theta_vals':
                # distancia euclídea que recorre el efector final, para este action step, en el config space
                
                # TODO: Implement this

                step_cost = 0

            elif ct == 'dist_to_goal_in_ws':
                # distancia euclídea que recorre el efector final, para este action step, en el workspace
                # TODO: Implement this
                step_cost = 0

            else:
                # por defecto: costo unitario
                step_cost = 1

        # 4) Acumular y devolver
        return c + step_cost

if __name__ == "__main__":

    # SCARA Robot SR-6iA dimensions
    # https://www.fanuc.eu/cz/cs/roboty/str%C3%A1nka-filtru-robot%C5%AF/scara-series/scara-sr-6ia

    length_vec = [0.35, 0.30]
    obstacles = [
        ((-0.1, -0.5), (0.1, -0.1)),
        ((1.00-0.9, -0.5), (1.25-0.9, -0.25)),
        ((0.25, 0.0), (0.5, 0.075)),
        ((-0.4, 0.0), (-0.2, 0.2)),
    ]
    init_pos = (-0.11, 0.38)
    goal_pos = (0.32, 0.111)
    angle_res = 15

    maze = SCARA_Robotic_Maze(
        length_vec,
        angle_resolution_in_degrees=angle_res,
        theta_0_range=[-np.pi, np.pi],
        theta_1_range=[-np.pi, np.pi],
        init_pos_of_gripper=init_pos,
        goal_pos=goal_pos,
        obstacles=obstacles
    )

    # para A estrella, no olvidar implementar straightline_dist del módulo gameiautils

    # descomentar cuando se hayan terminado las implementaciones requeridas

    #p = SCARA_Robotic_MazeSearchProblem(maze)
    #nsol, visited_nodes = graph_search(p, FIFOQueue())
    #print('Solucion BFS: {}. Nodos visitados={}. Costo Solucion = {}'.format(nsol.solution(), len(visited_nodes),nsol.path_cost))
    #displayResults(maze, visitedNodes=visited_nodes, solutionNodes=nsol.path())
    #display_pathCosts_ofResults(...) # terminar de implementar la función importada del módulo graphical_utils

    maze.show_spaces(maze.grid)