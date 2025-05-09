import numpy as np
import matplotlib.pyplot as plt
from scara_simulator_2D import Robot_SCARA

def plot_robot_movement(robot, theta_initial, delta_p,
                        show_desired_point=True,
                        show_links=True,
                        activate_grid=True):
    """
    Plot the SCARA robot movement in workspace and configuration space.

    For N=2: scatter plot of the two joint angles.
    For N>2: parallel-coordinates plot of all joint angles.

    Parameters
    ----------
    robot : Robot_SCARA
        Instance of Robot_SCARA with link lengths set.
    theta_initial : array-like of shape (n,)
        Initial joint angles [theta1, theta2, ..., thetan].
    delta_p : array-like of shape (2,)
        Desired displacement [dx, dy] of the end-effector.
    show_desired_point : bool, default=True
        Whether to display the desired end-effector location.
    show_links : bool, default=True
        Whether to plot the robot links.
    activate_grid : bool, default=True
        Whether to display the grid on plots.
    """
    theta_initial = np.array(theta_initial, dtype=float)
    delta_p = np.array(delta_p, dtype=float)

    # Compute initial end-effector position
    x_vec, y_vec, _, _, _ = robot.DirectKinematics(theta_initial)
    p0 = np.array([x_vec[-1], y_vec[-1]])

    # Compute Jacobian and pseudoinverse
    J = robot.computeJacobian(theta_initial)
    J_pinv = robot.computePseudoInverseJacobian(theta_initial)

    # Compute change in joint angles
    delta_theta = J_pinv @ delta_p

    # New joint angles and actual end-effector position
    theta_new = theta_initial + delta_theta
    x_vec_new, y_vec_new, _, _, _ = robot.DirectKinematics(theta_new)
    p_actual = np.array([x_vec_new[-1], y_vec_new[-1]])
    p_desired = p0 + delta_p

    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Workspace plot
    if show_links:
        coords_init = np.column_stack((np.insert(x_vec, 0, 0), np.insert(y_vec, 0, 0)))
        ax1.plot(coords_init[:, 0], coords_init[:, 1], color='blue', linestyle='--', linewidth=3)
        coords_new = np.column_stack((np.insert(x_vec_new, 0, 0), np.insert(y_vec_new, 0, 0)))
        ax1.plot(coords_new[:, 0], coords_new[:, 1], color='orange', linestyle='-', linewidth=3)

    ax1.scatter(p0[0], p0[1],   color='blue',  s=100, label='Initial')
    if show_desired_point:
        ax1.scatter(p_desired[0], p_desired[1], color='green', marker='*', s=300, label='Desired')
    ax1.scatter(p_actual[0], p_actual[1],   color='orange',s=100, label='Actual')
    ax1.scatter(0, 0, s=100, c='k')
    ax1.set_aspect('equal')
    ax1.set_title('Workspace')
    ax1.legend()
    if activate_grid:
        ax1.grid(True)

    # Configuration space
    n = robot.n
    if n == 2:
        ax2.scatter(theta_initial[0], theta_initial[1], color='blue',  s=100, label='Initial θ vals')
        ax2.scatter(theta_new[0],     theta_new[1],     color='orange',s=100, label='Actual θ vals')
        ax2.set_xlim(0, 2 * np.pi)
        ax2.set_ylim(0, 2 * np.pi)
        ax2.set_xlabel(r'$\theta_0$')
        ax2.set_ylabel(r'$\theta_1$')
    else:
        # Parallel-coordinates for all joints
        joint_indices = np.arange(n)
        ax2.plot(joint_indices, theta_new,     color='orange', marker='o', linestyle='-', linewidth=2, label='Actual θ vals')
        ax2.plot(joint_indices, theta_initial, color='blue',   marker='o', linestyle='--', linewidth=2, label='Initial θ vals')
        ax2.set_xticks(joint_indices)
        ax2.set_xticklabels([fr'$\theta_{{{i}}}$' for i in joint_indices])
        ax2.set_ylim(0, 2 * np.pi)
        # Y ticks at [0, π/2, π, 3π/2, 2π]
        yticks = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
        ylabels = ['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']
        ax2.set_yticks(yticks)
        ax2.set_yticklabels(ylabels)
    ax2.set_title('Configuration Space')
    ax2.legend()
    if activate_grid:
        ax2.grid(True)

    plt.tight_layout()
    plt.show()

def show_two_states():
    """
    Demonstrate a two-state movement of a N-link SCARA robot (N=7)

    This function:
    1. Sets a fixed random seed (20141719) for reproducibility.
    2. Constructs a SCARA robot with N links of decreasing length (each 10% shorter than the previous, starting at 1.0).
    3. Samples a random initial joint configuration.
    4. Defines a desired end-effector displacement.
    5. Calls `plot_robot_movement` to visualize both the initial and displaced states
       in the workspace and configuration space.

    This functions allows to visualize the effect of computing the IK, given an initial robot configuration,
    and a displacement from it (in the world space).
    
    Recall the approximation formula used: delta_theta = J_pinv @ delta_p,
    where:
        - J_pinv: is the pseudo-inverse of the Jacobian of the mechanism, calculated for a given set of initial angles.
        - delta_p: is the offset of the end-effector (in the world space)
        - delta_theta: is the offset of the theta vector (in the configuration space)

    """
    # For reproducibility
    np.random.seed(20141719)

    # Number of links
    N = 7 # test for N=2 and for N>2
    # Generate link lengths: first is 1.0, each subsequent is 10% shorter
    lengths = [1.0 * (0.9 ** i) for i in range(N)]

    # Instantiate SCARA robot
    robot = Robot_SCARA(lengths)

    # Random initial joint angles
    theta_initial = np.random.rand(N) * 2 * np.pi

    # Desired displacement
    delta_p = np.array([0.2, -0.1])*3

    # Plot movement
    plot_robot_movement(robot, theta_initial, delta_p)

if __name__ == '__main__':
    show_two_states()