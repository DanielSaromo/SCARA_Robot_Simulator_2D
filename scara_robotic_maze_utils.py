import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
from matplotlib.collections import LineCollection

from gameia_utils import Maze
from scara_simulator_2D import Robot_SCARA

from graphical_utils import convert_maze_toHeatmap

from matplotlib.animation import FuncAnimation, FFMpegWriter # for the video generation

class SCARA_Robotic_Maze(Maze):
    """
    Construye el maze en espacio de configuraciones ABSOLUTAS de un robot SCARA de dos eslabones.

    La clase base Maze puede usar 'E' o 'G' para marcar las celdas de salida.
    Este constructor marca primero las metas alcanzables como 'E', intenta
    construir el Maze; si falla porque no detecta 'E', reconvierte esas celdas
    a 'G' y vuelve a invocar al padre. Finalmente asegura que la celda de salida
    elegida quede como 'G'.
    """
    def __init__(self, length_vec, angle_resolution_in_degrees=0.5,
                 theta_0_range=[-np.pi, np.pi], theta_1_range=[-np.pi, np.pi],
                 obstacles=None, init_pos_of_gripper=None, goal_pos=(1.0, 1.0),
                 ws_tolerance=0.07):
        """Inicializa el maze y marca start (S), exit temporal (E/H) y final (G)."""
        assert len(length_vec) == 2, \
            "Por simplicidad, solo se permite que el robot tenga 2 eslabones."
        # Simulador SCARA
        self.robot_env = Robot_SCARA(length_vec)

        # workspace start & goal
        self.init_pos_of_gripper = init_pos_of_gripper or (0, sum(length_vec))
        if isinstance(goal_pos, tuple):
            self.goal_position = goal_pos
        elif isinstance(goal_pos, list) and len(goal_pos) == 1:
            self.goal_position = goal_pos[0]
        else:
            raise ValueError("goal_pos debe ser tupla (x,y) o lista de un elemento.")
        gx, gy = self.goal_position

        self.ws_tolerance = ws_tolerance

        # obstáculos
        default_obs = [
            ((0.5, 1.00), (0.75, 1.5)),
            ((0.25, 0.5), (0.5, 1.00)),
            ((1.00, -0.5), (1.25, -0.25)),
            ((1.25, -0.5), (2.0, 0.5)),
        ]
        self.obstacles = obstacles or default_obs

        # grid inicial (~ = unexplorado)
        grid = self.create_SCARA_grid(angle_resolution_in_degrees,
                                      theta_0_range,
                                      theta_1_range)

        # --- Start: dos soluciones IK ---
        down, up = self.robot_env.InverseKinematics_2Links(
            [self.init_pos_of_gripper[0]],
            [self.init_pos_of_gripper[1]],
            returnAllSolutions=True
        )
        self.startCells = []
        for q1, q2 in (down, up):
            a0 = q1[0]; a1 = a0 + q2[0]
            # convert to grid indices
            r, c = self.angles_to_index(a0, a1)
            # check if that configuration is collision-free
            rel0, rel1 = self.fromAbsoluteAngles_toRelativeAngles(a0, a1)
            feasible = self.isValidState((rel0, rel1))
            # mark as 'S' if OK, otherwise as 'Z'
            grid[r][c] = 'S' if feasible else 'Z'
            self.startCells.append((r, c))

        # —— VERIFY start reachability within tolerance —— 
        start_errors = []
        for (r, c) in self.startCells:
            x, y = self.config_to_world(r, c)
            err = np.linalg.norm([x - self.init_pos_of_gripper[0],
                                  y - self.init_pos_of_gripper[1]])
            start_errors.append(err)
        if min(start_errors) > self.ws_tolerance:
            print(f"⚠️ Warning: requested start EE-position "
                  f"{self.init_pos_of_gripper!r} not reached (min error={min(start_errors):.3f} m). "
                  "Please choose a different initial point.")

        # --- Goal: dos soluciones IK, marcamos 'E' si es factible, 'H' si no ---
        down_g, up_g = self.robot_env.InverseKinematics_2Links(
            [gx], [gy], returnAllSolutions=True
        )
        self.exitCells = []
        for q1, q2 in (down_g, up_g):
            a0 = q1[0]; a1 = a0 + q2[0]
            r, c = self.angles_to_index(a0, a1)
            rel0, rel1 = self.fromAbsoluteAngles_toRelativeAngles(a0, a1)
            feasible = self.isValidState((rel0, rel1))
            grid[r][c] = 'G' if feasible else 'H'
            self.exitCells.append((r, c))

        #self.exitCell = self.exitCells[0] if self.exitCells else None

        # —— VERIFY goal reachability within tolerance —— 
        exit_errors = []
        for (r, c) in self.exitCells:
            x, y = self.config_to_world(r, c)
            err = np.linalg.norm([x - self.goal_position[0],
                                  y - self.goal_position[1]])
            exit_errors.append(err)
        if min(exit_errors) > self.ws_tolerance:
            print(f"⚠️ Warning: desired goal EE-position "
                  f"{self.goal_position!r} not reached (min error={min(exit_errors):.3f} m). "
                  "Please choose a different goal point.")

        # Intentar construir con 'G' como salida
        try:
            super().__init__(grid)
        except AttributeError:
            # Falló: no detectó 'E'; reconvertir 'E' a 'G'
            new_grid = [row.copy() for row in grid]
            for r, c in self.exitCells:
                if new_grid[r][c] == 'E':
                    new_grid[r][c] = 'G'
            super().__init__(new_grid)

        # Volver a cargar el grid original (con la marca 'G') en self.grid
        self.grid = [row.copy() for row in grid]
        # Asegurar que la celda de salida quede como 'G'
        er, ec = self.exitCell
        self.grid[er][ec] = 'G'

        # actualizar colisiones en el resto de la grilla
        self.updateGrid()

    # --- Funciones auxiliares internas ---
    def index_to_angles(self, row, col):
        """De índices (r,c) devuelve los ángulos absolutos θ0, θ1."""
        return self.theta0_vals[col], self.theta1_vals[row]

    def angles_to_index(self, abs0, abs1):
        """Normaliza ambos ángulos a [–π,π] y devuelve los índices de grilla más cercanos."""
        # -------- normalizamos --------
        abs0 = ((abs0 + np.pi) % (2*np.pi)) - np.pi
        abs1 = ((abs1 + np.pi) % (2*np.pi)) - np.pi
        # -------------------------------

        col = int(np.argmin(np.abs(self.theta0_vals - abs0)))
        row = int(np.argmin(np.abs(self.theta1_vals - abs1)))
        return row, col

    def config_to_world(self, row, col):
        """Dado un estado (r,c) retorna la posición (x,y) del end-effector."""
        abs0, abs1 = self.index_to_angles(row, col)
        rel0, rel1 = self.fromAbsoluteAngles_toRelativeAngles(abs0, abs1)
        x_vec, y_vec, *_ = self.robot_env.DirectKinematics([rel0, rel1])
        return np.array([x_vec[-1], y_vec[-1]])
    # -------------------------------------

    def __str__(self, flip_horizontal_print=False):

        if flip_horizontal_print:
            header = ' ' + '-'*self.numCols + ' '
            lines = [header]
            # Row 0 should be at the bottom, so we walk the rows in reverse:
            for row in reversed(self.grid):
                lines.append('|' + ''.join(row) + '|')
            lines.append(header)
            return '\n'.join(lines)

        else:
            header = ' ' + '-'*self.numCols + ' '
            body   = ['|' + ''.join(row) + '|' for row in self.grid]
            return '\n'.join([header] + body + [header])

    def isValidState(self, theta_vec):
        rel0, rel1 = theta_vec
        x, y, _, _, _ = self.robot_env.DirectKinematics([rel0, rel1])
        p0 = (0,0); p1 = (x[0], y[0]); p2 = (x[1], y[1])
        for ll, ur in self.obstacles:
            rect = (ll[0], ll[1], ur[0], ur[1])
            if self._segment_intersects_rect(p0, p1, rect): return False
            if self._segment_intersects_rect(p1, p2, rect): return False
        return True

    def isBlocked(self, row, col):
        cur = self.grid[row][col]
        if cur != '~': return cur == '#'
        a0, a1 = self.index_to_angles(row, col)
        rel0, rel1 = self.fromAbsoluteAngles_toRelativeAngles(a0, a1)
        ok = self.isValidState((rel0, rel1))
        self.grid[row][col] = ' ' if ok else '#'
        return not ok

    def fromAbsoluteAngles_toRelativeAngles(self, a0, a1):
        return a0, a1 - a0

    def fromRelativeAngles_toAbsoluteAngles(self, rel0, rel1):
        # rel0, rel1 son ambos relativos (lo que devolvería la función de arriba (de abs a rel))
        abs0 = rel0
        abs1 = rel0 + rel1
        return abs0, abs1


    def updateGrid(self):
        skip = set()
        for r in range(self.numRows):
            for c in range(self.numCols):
                if self.grid[r][c] in ('S', 'Z', 'G', 'H'):
                    skip.add((r, c))

        for r in range(self.numRows):
            for c in range(self.numCols):
                if (r,c) in skip: continue
                self.isBlocked(r, c)

    def _segment_intersects_rect(self, p1, p2, rect):
        x_min, y_min, x_max, y_max = rect
        for px, py in (p1, p2):
            if x_min <= px <= x_max and y_min <= py <= y_max:
                return True
        edges = [
            ((x_min,y_min),(x_min,y_max)),
            ((x_min,y_max),(x_max,y_max)),
            ((x_max,y_max),(x_max,y_min)),
            ((x_max,y_min),(x_min,y_min)),
        ]
        for q1, q2 in edges:
            if self._segments_intersect(p1, p2, q1, q2): return True
        return False

    def _segments_intersect(self, p1, p2, q1, q2):
        def orient(a,b,c):
            v = (b[1]-a[1])*(c[0]-b[0]) - (b[0]-a[0])*(c[1]-b[1])
            if abs(v)<1e-9: return 0
            return 1 if v>0 else 2
        def on_seg(a,b,c):
            return (min(a[0],c[0])<=b[0]<=max(a[0],c[0]) and
                    min(a[1],c[1])<=b[1]<=max(a[1],c[1]))
        o1 = orient(p1,p2,q1); o2 = orient(p1,p2,q2)
        o3 = orient(q1,q2,p1); o4 = orient(q1,q2,p2)
        if o1!=o2 and o3!=o4: return True
        if o1==0 and on_seg(p1,q1,p2): return True
        if o2==0 and on_seg(p1,q2,p2): return True
        if o3==0 and on_seg(q1,p1,q2): return True
        if o4==0 and on_seg(q1,p2,q2): return True
        return False

    def _closest_index(self, angle, angle_list):
        return int(np.argmin(np.abs(angle_list - angle)))

    def create_SCARA_grid(self, angle_resolution_in_degrees, theta_0_range, theta_1_range):
        res = angle_resolution_in_degrees * np.pi/180.0
        self.theta0_vals = np.arange(theta_0_range[0],
                                     theta_0_range[1] + res/2, res)
        self.theta1_vals = np.arange(theta_1_range[0],
                                     theta_1_range[1] + res/2, res)
        self.numCols = len(self.theta0_vals)
        self.numRows = len(self.theta1_vals)
        return [['~' for _ in range(self.numCols)]
                for _ in range(self.numRows)]

    def show_spaces(self, grid, lightred_color = '#e29e9e', lightgreen_color = '#abd293', lightgray_color = 'lightgray'):
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,6),dpi=100)

        # --- World Space ---
        for ll, ur in self.obstacles:
            w, h = ur[0] - ll[0], ur[1] - ll[1]
            ax1.add_patch(Rectangle(ll, w, h, fill=True, color=lightgray_color))

        # --- Punto en el origen (0,0) ---
        ax1.scatter(0, 0, c='black', s=100)

        ax1.scatter(*self.init_pos_of_gripper,
                    c='red', marker='o', label='Start')
        ax1.scatter(*self.goal_position,
                    c='green', marker='o', label='Goal')

        # Draw both IK-goal solutions:
        # solid line if feasible, dotted if not
        down_g, up_g = self.robot_env.InverseKinematics_2Links(
            [self.goal_position[0]], [self.goal_position[1]],
            returnAllSolutions=True)
        for q_rel in (down_g, up_g):
            q1, q2 = q_rel
            a0 = q1[0]; a1 = a0 + q2[0]
            rel0, rel1 = self.fromAbsoluteAngles_toRelativeAngles(a0, a1)
            xs, ys, _, _, _ = self.robot_env.DirectKinematics([rel0, rel1])
            feasible = self.isValidState((rel0, rel1))
            ls = '-' if feasible else ':'
            ax1.plot([0, xs[0]], [0, ys[0]], ls, color='orange', linewidth=2)
            ax1.plot([xs[0], xs[1]], [ys[0], ys[1]], ls, color='purple', linewidth=2)

        # Dibuja AMBAS soluciones del robot para la posición inicial (codo arriba y codo abajo)
        for r_s, c_s in self.startCells:
            a0_s, a1_s = self.index_to_angles(r_s, c_s)
            rel0_s, rel1_s = self.fromAbsoluteAngles_toRelativeAngles(a0_s, a1_s)
            xs, ys, _, _, _ = self.robot_env.DirectKinematics([rel0_s, rel1_s])
            feasible = self.isValidState((rel0_s, rel1_s))
            ls = '--' if feasible else '-.'  # Dashed para factible, dash-dot para no factible
            ax1.plot([0, xs[0]], [0, ys[0]], ls, color='orange', linewidth=2)
            ax1.plot([xs[0], xs[1]], [ys[0], ys[1]], ls, color='purple', linewidth=2)

        ax1.set_title('World Space')
        ax1.set_aspect('equal')
        ax1.set_xlabel('X'); ax1.set_ylabel('Y')
        ax1.legend()

        # --- Configuration Space ---
        nrows, ncols = len(grid), len(grid[0])
        data = np.zeros((nrows, ncols), dtype=int)
        # este es el mapping de colores de celda, para cada caracter
        char2idx = {
                        '#': 0,   # celda bloqueada
                        'S': 1,   # start factible
                        'G': 2,   # goal factible
                        ' ': 3,   # celda libre
                        '~': 4,   # no explorada
                        'Z': 0,   # start no factible
                        'H': 0,   # goal no factible
                    }

        for r in range(nrows):
            for c in range(ncols):
                data[r, c] = char2idx.get(grid[r][c], 4) # el default es el numero asignado a la celda no explorada

        # esta variable está relacionada con char2idx. nota que en la variable char2idx, Z y H, tienen color negro asociado
        cmap = ListedColormap([lightgray_color,'red','green','white','gray'])
        ax2.imshow(data, origin='upper', cmap=cmap, norm=mcolors.NoNorm(), interpolation='none')
        self.data = data

        # Overlay start & goal markers (we only put markers on the unfeasible cells)
        for r, c in self.startCells:
            cell_mark = grid[r][c]
            if cell_mark == 'Z':
                ax2.scatter(c, r, marker='X', color=lightred_color, linewidth=0.25)
        for r, c in self.exitCells:
            cell_mark = grid[r][c]
            if cell_mark == 'H':
                ax2.scatter(c, r, marker='X', color=lightgreen_color, linewidth=0.25)

        # Angle ticks
        ticks = np.arange(-np.pi, np.pi + 1e-6, np.pi/2)
        # Para cada tick, angles_to_index devuelve (row, col);
        # la columna es el índice en theta0, la fila en theta1
        xt = [ self.angles_to_index(t, t)[1] for t in ticks ]
        yt = [ self.angles_to_index(t, t)[0] for t in ticks ]
        labels = ['-π','-π/2','0','π/2','π']
        ax2.set_xticks(xt); ax2.set_xticklabels(labels)
        ax2.set_yticks(yt); ax2.set_yticklabels(labels)

        # Legend for configuration space
        import matplotlib.patches as mpatches
        from matplotlib.lines import Line2D
        legend_handles = [
            mpatches.Patch(facecolor='white', edgecolor='black', label='Feasible state'),
            mpatches.Patch(facecolor=lightgray_color, label='Collision'),
            mpatches.Patch(facecolor='red', label='Feasible start'),
            Line2D([0],[0], marker='x', color=lightred_color, linestyle='None', label='Unfeasible start'),
            mpatches.Patch(facecolor='green', label='Feasible goal'),
            Line2D([0],[0], marker='x', color=lightgreen_color, linestyle='None', label='Unfeasible goal'),
        ]
        #ax2.legend(handles=legend_handles, loc='upper right', title='Config Space Legend', ncol=3)
        ax2.legend(handles=legend_handles, loc='upper right', ncol=3)

        ax2.set_title('Configuration Space')
        ax2.set_xlabel('θ_abs1'); ax2.set_ylabel('θ_abs2')

        plt.tight_layout()
        plt.savefig('scara_ws_and_cs.pdf', format='pdf', bbox_inches='tight')  # Save the figure as a PDF
        plt.show()

    def show_spaces_and_robotPath(self, grid, actions, initial_state=None, lightblue_color = '#00fff0', color_cfg_traj=True, debugVerbose=False):
        """
        Muestra:
          - World space con obstáculos, configuración inicial real (dashed) y punto inicial (rojo).
          - Trayectorias del end-effector (EE) y del joint medio con sus leyendas.
          - Punto goal al final.
          - Config space con fondo de show_spaces y marcaje de estados no factibles con X 
            (colores originales para Z/H).
        """

        # Preparamos figura
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=100)

        # --- WORLD SPACE: obstáculos ---
        for ll, ur in self.obstacles:
            w, h = ur[0] - ll[0], ur[1] - ll[1]
            ax1.add_patch(Rectangle(ll, w, h, fill=True, color='lightgray'))

        # --- Punto en el origen (0,0) ---
        ax1.scatter(0, 0, c='black', s=100,)

        # --- Configuración inicial real (dashed) y punto inicial (rojo) ---
        # if caller gave one, use it; otherwise fall back to using the first one
        if initial_state is not None:
            initial = initial_state
        else:
            feasible = [(r,c) for (r,c) in self.startCells if grid[r][c]=='S']
            initial = feasible[0] if feasible else self.startCells[0]

        # now `initial` is always the one you pass in
        r0, c0 = initial
        abs0_i, abs1_i = self.index_to_angles(r0, c0)
        rel0_i, rel1_i = self.fromAbsoluteAngles_toRelativeAngles(abs0_i, abs1_i)
        xs_i, ys_i, *_ = self.robot_env.DirectKinematics([rel0_i, rel1_i])

        # enlaces discontinua
        ax1.plot([0, xs_i[0]], [0, ys_i[0]], '--', color='orange', linewidth=2)
        ax1.plot([xs_i[0], xs_i[1]], [ys_i[0], ys_i[1]], '--', color='purple', linewidth=2)
        # punto inicial
        ax1.scatter(xs_i[-1], ys_i[-1], c='red', s=50, label='Start')

        # Helper de acciones
        deltas = {'N':(-1,0), 'S':( 1,0), 'E':(0, 1), 'W':(0,-1),
          'NE':(-1,1), 'NW':(-1,-1), 'SE':(1,1), 'SW':(1,-1)}
        def apply_action(state, action):
            if isinstance(action, tuple) and action[0]=='J': return action[1]
            if action in deltas:
                dr, dc = deltas[action]; r, c = state
                return ((r+dr)%self.numRows, (c+dc)%self.numCols)
            return state

        # Secuencia de estados
        state0 = initial
        path_states = [state0]
        for a in actions:
            state0 = apply_action(state0, a)
            path_states.append(state0)

        # Coordenadas de trayectorias
        ee_xs, ee_ys, mid_xs, mid_ys = [], [], [], []
        for r, c in path_states:
            abs0, abs1 = self.index_to_angles(r, c)
            rel0, rel1 = self.fromAbsoluteAngles_toRelativeAngles(abs0, abs1)
            xs, ys, *_ = self.robot_env.DirectKinematics([rel0, rel1])
            mid_xs.append(xs[0]); mid_ys.append(ys[0])
            ee_xs.append(xs[1]); ee_ys.append(ys[1])
        
        # Plot trayectorias
        ax1.plot(ee_xs, ee_ys, '-', color='blue', linewidth=1, label='Trayectoria del EE')
        ax1.plot(mid_xs, mid_ys, '-', color=lightblue_color, linewidth=1, label='Trayectoria del middle joint')
        ax1.scatter(ee_xs, ee_ys, c='blue', s=30)
        ax1.scatter(mid_xs, mid_ys, c=lightblue_color, s=30)

        # Punto goal final
        ax1.scatter(*self.goal_position, c='green', marker='o', s=50, label='Goal')

        # Ajustes world space
        ax1.set_title('World Space con trayectoria')
        ax1.set_aspect('equal'); ax1.set_xlabel('X'); ax1.set_ylabel('Y')
        ax1.legend(loc='upper right')

        # --- CONFIGURATION SPACE ---
        nrows, ncols = len(grid), len(grid[0])
        data = np.zeros((nrows, ncols), dtype=int)
        char2idx = {'#':0,'S':1,'G':2,' ':3,'~':4,'Z':0,'H':0}
        for r in range(nrows):
            for c in range(ncols): data[r,c] = char2idx.get(grid[r][c],4)
        cmap = ListedColormap(['lightgray','red','green','white','gray'])
        ax2.imshow(data, origin='upper', cmap=cmap, norm=mcolors.NoNorm(), interpolation='none')

        # Dibujar trayectoria en C-space, gradient o sólido
        cs_x = [c for r,c in path_states]
        cs_y = [r for r,c in path_states]
        if color_cfg_traj:
            points = np.array([cs_x, cs_y]).T.reshape(-1,1,2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            cmap = cm.get_cmap('RdYlGn')
            norm = mcolors.Normalize(vmin=0, vmax=len(segments)-1)
            lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2)
            lc.set_array(np.arange(len(segments)))
            ax2.add_collection(lc)
        else:
            ax2.plot(cs_x, cs_y, '-', linewidth=2)

        # X en estados no factibles según show_spaces
        lightred, lightgreen = '#e29e9e', '#abd293'
        for r_s, c_s in self.startCells:
            if grid[r_s][c_s]=='Z': ax2.scatter(c_s, r_s, marker='X', color=lightred, linewidth=0.5)
        for r_g, c_g in self.exitCells:
            if grid[r_g][c_g]=='H': ax2.scatter(c_g, r_g, marker='X', color=lightgreen, linewidth=0.5)

        # Ticks angulares y legend intacta
        ticks = np.arange(-np.pi, np.pi+1e-6, np.pi/2)
        xt = [self.angles_to_index(t,t)[1] for t in ticks]; yt = [self.angles_to_index(t,t)[0] for t in ticks]
        labels = ['-π','-π/2','0','π/2','π']
        ax2.set_xticks(xt); ax2.set_xticklabels(labels)
        ax2.set_yticks(yt); ax2.set_yticklabels(labels)
        ax2.set_title('Configuration Space con trayectoria')
        ax2.set_xlabel('Índice θ0 (col)'); ax2.set_ylabel('Índice θ1 (row)')

        if debugVerbose:
            # DEBUG: list all start-cells and which one we’re picking
            print("  [DEBUG] maze.startCells:", self.startCells)
            marks = [grid[r][c] for (r,c) in self.startCells]
            print("  [DEBUG] grid marks at startCells:", marks)
            feasible_starts = [(r, c) for (r, c) in self.startCells if grid[r][c] == 'S']
            print("  [DEBUG] feasible_starts:", feasible_starts)
            print("  [DEBUG] show_spaces chosen initial:", initial)

        plt.tight_layout()
        plt.savefig('path_scara_ws_and_cs.pdf', bbox_inches='tight')
        plt.show()

    def animate_robot_path(self, grid, actions, initial_state=None, output_file='trajectory.mp4',
                            fps=10, color_cfg_traj=True):
        """
        Animate the robot moving along `actions`, starting from initial_state if given, saving an MP4 with:
         - world space (obstacles, static full path, and only the current robot drawn)
         - configuration space (static full trajectory, with current config shown as a black dot)
         If color_cfg_traj is True, the C-space path is drawn with a RdYlGn gradient.
        """

        # 1) Helper to apply an action in C-space
        deltas = {'N':(-1,0), 'S':( 1,0), 'E':(0, 1), 'W':(0,-1),
          'NE':(-1,1), 'NW':(-1,-1), 'SE':(1,1), 'SW':(1,-1)}
        def apply_action(state, action):
            if isinstance(action, tuple) and action[0]=='J':
                return action[1]
            dr, dc = deltas.get(action, (0,0))
            r, c = state
            return ((r+dr) % self.numRows, (c+dc) % self.numCols)

        # 2) Pick the exact start you passed in, or fall back to the first feasible
        if initial_state is not None:
            initial = initial_state
        else:
            feasible = [(r,c) for (r,c) in self.startCells if grid[r][c]=='S']
            initial = feasible[0] if feasible else self.startCells[0]

        # 3) Build the full path of states
        path = [initial]
        for a in actions:
            path.append(apply_action(path[-1], a))

        # 4) Precompute world-space & C-space coords
        ee_xs, ee_ys, mid_xs, mid_ys = [], [], [], []
        cs_x, cs_y = [], []
        for (r, c) in path:
            abs0, abs1 = self.index_to_angles(r, c)
            rel0, rel1 = self.fromAbsoluteAngles_toRelativeAngles(abs0, abs1)
            xs, ys, *_ = self.robot_env.DirectKinematics([rel0, rel1])
            mid_xs.append(xs[0]); mid_ys.append(ys[0])
            ee_xs.append(xs[1]); ee_ys.append(ys[1])
            cs_x.append(c); cs_y.append(r)

        # 5) Set up figure and static artists
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,6), dpi=100)

        # --- Static World Space ---
        for ll, ur in self.obstacles:
            w, h = ur[0]-ll[0], ur[1]-ll[1]
            ax1.add_patch(Rectangle(ll, w, h, fill=True, color='lightgray'))
        ax1.scatter(0,0,c='black',s=100)
        ax1.scatter(*self.goal_position, c='green', marker='o', s=50, label='Goal')
        ax1.plot(ee_xs, ee_ys, '-', linewidth=1, label='EE path')
        ax1.plot(mid_xs, mid_ys, '-', linewidth=1, label='Middle path')
        ax1.set_aspect('equal'); ax1.set_xlabel('X'); ax1.set_ylabel('Y')
        ax1.set_title('World Space'); ax1.legend(loc='upper right')

        # --- Static Config Space ---
        nrows, ncols = len(grid), len(grid[0])
        data = np.zeros((nrows,ncols),dtype=int)
        char2idx = {'#':0,'S':1,'G':2,' ':3,'~':4,'Z':0,'H':0}
        for i in range(nrows):
            for j in range(ncols):
                data[i,j] = char2idx.get(grid[i][j], 4)
        cmap_cfg = ListedColormap(['lightgray','red','green','white','gray'])
        ax2.imshow(data, origin='upper', cmap=cmap_cfg,
                   norm=mcolors.NoNorm(), interpolation='none')

        # --- Plot C-space trajectory, either gradient or solid ---
        if color_cfg_traj:
            # build segments for LineCollection
            points = np.array([cs_x, cs_y]).T.reshape(-1,1,2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            cmap = cm.get_cmap('RdYlGn')
            norm = mcolors.Normalize(vmin=0, vmax=len(segments)-1)
            lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2)
            lc.set_array(np.arange(len(segments)))
            ax2.add_collection(lc)
        else:
            ax2.plot(cs_x, cs_y, '-', linewidth=2)

        # angular ticks
        ticks = np.arange(-np.pi, np.pi+1e-6, np.pi/2)
        xt = [ self.angles_to_index(t,t)[1] for t in ticks ]
        yt = [ self.angles_to_index(t,t)[0] for t in ticks ]
        labels = ['-π','-π/2','0','π/2','π']
        ax2.set_xticks(xt); ax2.set_xticklabels(labels)
        ax2.set_yticks(yt); ax2.set_yticklabels(labels)
        ax2.set_title('Configuration Space'); 
        ax2.set_xlabel('θ0 idx'); ax2.set_ylabel('θ1 idx')

        # 6) Dynamic artists for animation
        link1_line, = ax1.plot([],[], '-', color='orange', linewidth=3)
        link2_line, = ax1.plot([],[], '-', color='purple', linewidth=3)
        cs_dot = ax2.scatter([], [], c='black', s=50, zorder=5)

        def update(frame):
            # world-space robot links
            x0, y0 = 0, 0
            x1, y1 = mid_xs[frame], mid_ys[frame]
            x2, y2 = ee_xs[frame], ee_ys[frame]
            link1_line.set_data([x0, x1], [y0, y1])
            link2_line.set_data([x1, x2], [y1, y2])
            # C-space current config
            cs_dot.set_offsets([[cs_x[frame], cs_y[frame]]])
            return link1_line, link2_line, cs_dot

        # 7) Create and save animation
        anim = FuncAnimation(fig, update, frames=len(path),
                             blit=True, interval=1000/fps)
        writer = FFMpegWriter(fps=fps)
        anim.save(output_file, writer=writer)
        plt.close(fig)

    def printGrid(self):
        print(self)

if __name__ == "__main__":

    # be careful when putting the new goal positions! they need to be reachable!

    str_of_case_to_use = 'B'
    
    if str_of_case_to_use == 'A':
        length_vec = [1.0, 0.5]

        obstacles = [
            ((0.5, 1.00), (0.75, 1.5)),
            ((-1, -0.5), (-0.75, -0.25)),
            ((1.00, -0.5), (1.25, -0.25)),
            ((1.25, -0.5), (2.0, 0.5)),
        ]

        maze = SCARA_Robotic_Maze(
            length_vec,
            angle_resolution_in_degrees=5,
            theta_0_range=[-np.pi, np.pi],
            theta_1_range=[-np.pi, np.pi],
            init_pos_of_gripper=(0, 1.5),
            obstacles=obstacles,
            #goal_pos=(1, 1)
            goal_pos=(0.9, 1.1)
        )

    elif str_of_case_to_use == 'B':
        # SCARA Robot SR-6iA
        # https://www.fanuc.eu/cz/cs/roboty/str%C3%A1nka-filtru-robot%C5%AF/scara-series/scara-sr-6ia
        length_vec = [0.35, 0.30]

        obstacles = [
            ((-0.1, -0.5), (0.1, -0.1)),
            #((0.25, 0.5), (0.5, 0.3)),
            ((1.00-0.9, -0.5), (1.25-0.9, -0.25)),
            ((0.25, 0.0), (0.5, 0.075)),
            ((-0.4, 0.0), (-0.2, 0.2)),
        ]

        pos_inicial = (-0.11, 0.38)
        pos_objetivo = (0.32, 0.111)

        maze = SCARA_Robotic_Maze(
            length_vec, obstacles=obstacles,
            angle_resolution_in_degrees=15,
            theta_0_range=[-np.pi, np.pi],
            theta_1_range=[-np.pi, np.pi],
            init_pos_of_gripper=pos_inicial,
            goal_pos=pos_objetivo,
        )

    # prints the ASCII grid representation of the config space
    maze.printGrid()

    # show the world space and config space, for the initial and goal states of the robot
    maze.show_spaces(maze.grid)

    # show the config space as a pure grid search environment
    convert_maze_toHeatmap(maze.grid)

    seq_of_actions = ['N', 'E', 'N', 'E', 'N', 'N', 'N', 'N', 'N', 'N'] # example of list of actions in the config space
    initial_config_state = maze.startCells[0]

    # show the world space and config space, and the trajectory generated by the sequence of actions
    maze.show_spaces_and_robotPath(maze.grid, seq_of_actions, initial_state=initial_config_state)

    # creates a video of the world space and config space, for the trajectory generated by the sequence of actions
    maze.animate_robot_path(
        maze.grid,
        actions = seq_of_actions,
        initial_state=initial_config_state,
        output_file="video_scara.mp4",
        fps=15
    )