# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class Robot_SCARA():
    def __init__(self, length_vec):
        """
        Initialize a SCARA robot model.

        Parameters
        ----------
        length_vec : array-like
            Link lengths [l1, l2, ..., ln].

        Attributes
        ----------
        length_vec : np.ndarray
            Array of link lengths.
        n : int
            Number of links/joints.
        x_0, y_0 : float
            Base coordinates (origin).
        x, y : pd.DataFrame or None
            Stored input joint angles and output kinematics DataFrame.
        add_noise : bool
            Flag to add noise to joint angles in DirectKinematics.
        perc_noise : float
            Noise scaling percentage.
        random_state : int
            Seed for reproducibility.
        dimension_limit_inf, dimension_limit_sup : float or None
            Bounds for sampling joint angles when generating datasets.
        jacobian : np.ndarray or None
            Last computed 2×2 Jacobian matrix.
        pseudoInvOf_jacobian : np.ndarray or None
            Last computed pseudoinverse of the Jacobian.
        angleVector_toEachLinkEnd : np.ndarray or None
            Angle of the vector from the origin to each link end (via arctan2).
        """
        self.length_vec = np.array(length_vec)
        self.n = len(self.length_vec)
        self.x_0 = 0
        self.y_0 = 0

        self.x = None
        self.y = None

        self.add_noise = False
        self.perc_noise = 0.01 #percentage of multiplicative noise for the relative angles

        self.random_state = 20141719

        self.dimension_limit_inf = None
        self.dimension_limit_sup = None

        self.jacobian = None
        self.pseudoInvOf_jacobian = None

        self.angleVector_toEachLinkEnd = None

    def DirectKinematics(self, theta_rel_vec):
        """
        Compute direct kinematics for the SCARA robot.

        This method calculates, for each link:
        - X and Y end positions
        - Absolute joint orientation (sum of relative angles)
        - Orientation of the position vector from the base to each link end
        - Radial distance from the base to each link end
        - Now we output the absolute angle vector and the angle vector from the origin to the end part of the links

        Parameters
        ----------
        theta_rel_vec : array-like of shape (n,)
            Relative joint angles [theta1, theta2, ..., thetan].

        Returns
        -------
        x_vec : np.ndarray, shape (n,)
            X coordinates of each link end.
        y_vec : np.ndarray, shape (n,)
            Y coordinates of each link end.
        theta_abs_vec : np.ndarray, shape (n,)
            Cumulative joint orientations (absolute angles of each joint).
        angle_vec : np.ndarray, shape (n,)
            Angle of the vector from the origin to each link tip (arctan2 of y_vec, x_vec).
        r_vec : np.ndarray, shape (n,)
            Radial distances from the base origin to each link end.
        """
        theta_rel = np.array(theta_rel_vec, dtype=float)
        # Optionally add random noise to joint angles
        if self.add_noise:
            theta_rel += (np.random.rand(self.n) - 0.5) * 2 * self.perc_noise

        # Construct matrix to sum relative angles into absolute orientations
        grouping_matrix = np.transpose(np.tril(np.ones((self.n, self.n))))
        # Absolute joint angles: each element is sum of all preceding relative angles
        # ejm: for theta_rel_vec = [theta_1, theta_2, theta_3],
        #      then theta_abs_vec = [theta_1, theta_1 + theta_2, theta_1 + theta_2 + theta_3]
        self.theta_abs_vec  =   np.matmul(  theta_rel_vec  ,  grouping_matrix   )

        # Compute link end positions using absolute angles
        self.x_vec = self.x_0 + np.matmul(  self.length_vec*np.cos(self.theta_abs_vec)  ,  grouping_matrix   )
        self.y_vec = self.x_0 + np.matmul(  self.length_vec*np.sin(self.theta_abs_vec)  ,  grouping_matrix   )

        # Angle of the vector from base origin to each link end
        self.angleVector_toEachLinkEnd = np.arctan2(self.y_vec, self.x_vec)

        # Distance from base to each link end
        self.r_vec = np.sqrt(self.x_vec**2 + self.y_vec**2)

        return self.x_vec, self.y_vec, self.theta_abs_vec, self.angleVector_toEachLinkEnd, self.r_vec

    def InverseKinematics_2Links(self, x_finEff_vec, y_finEff_vec, returnAllSolutions=False):
        """
        Solve inverse kinematics analytically for a 2-link planar SCARA.

        Given desired end-effector coordinates, compute joint angles that achieve those positions.

        Parameters
        ----------
        x_finEff_vec, y_finEff_vec : array-like of shape (N,)
            Desired X and Y coordinates for the end-effector.
        returnAllSolutions : bool, default=False
            If False, return only elbow-down solution.
            If True, return both elbow-down and elbow-up branches.

        Returns
        -------
        If returnAllSolutions=False:
            q1_down, q2_down : np.ndarray of shape (N,)
        Else:
            [(q1_down, q2_down), (q1_up, q2_up)]
        """
        if len(self.length_vec) != 2:
            raise ValueError("This function only works for a robot with 2 links!")

        a1, a2 = self.length_vec
        q1_down, q2_down = [], []
        q1_up,   q2_up   = [], []

        for x, y in zip(x_finEff_vec, y_finEff_vec):
            cos_phi = (x**2 + y**2 - a1**2 - a2**2) / (2 * a1 * a2)
            cos_phi = np.clip(cos_phi, -1.0, 1.0)

            phi = np.arccos(cos_phi)
            q2a, q2b =  phi, -phi

            base_angle = np.arctan2(y, x)
            # compute shoulder angles for each elbow solution
            # q1 is recomputed according to each of the values for q2 that are plausible to obtain
            # hence, we obtain 2 solution pairs: (q1a, q2a) and (q1b, q2b)
            q1a = base_angle - np.arctan2(a2 * np.sin(q2a), a1 + a2 * np.cos(q2a))
            q1b = base_angle - np.arctan2(a2 * np.sin(q2b), a1 + a2 * np.cos(q2b))

            q1_down.append(q1a)
            q2_down.append(q2a)
            q1_up.append(q1b)
            q2_up.append(q2b)

        q1_down = np.array(q1_down)
        q2_down = np.array(q2_down)
        q1_up   = np.array(q1_up)
        q2_up   = np.array(q2_up)

        if returnAllSolutions:
            return [
                (q1_down, q2_down),  # elbow-down branch
                (q1_up,   q2_up)     # elbow-up branch
            ]
        else:
            return q1_down, q2_down

    def DirectKinematics_DataFrame(self, theta_rel_vec):
        """
        Return direct kinematics results as a pandas DataFrame.

        This now includes:
        - x_i, y_i               : link end positions
        - theta_abs_i            : absolute joint angles
        - angleVector_i          : vector orientation angles from base to each link end
        - r_i                    : radial distances

        Parameters
        ----------
        theta_rel_vec : array-like of shape (n,) or (N,n)
            One or more sets of relative joint angles.

        Returns
        -------
        df : pandas.DataFrame
            Combined input angles and kinematic outputs,
            with one row per input sample.
        """
        # ensure array form
        theta_arr = np.array(theta_rel_vec, dtype=float)
        # if a single n-vector was passed, reshape to (1,n)
        if theta_arr.ndim == 1:
            theta_arr = theta_arr.reshape(1, -1)
        # final check: must have n columns
        assert theta_arr.shape[1] == self.n, (
            f"Expected input shape (n,) or (N,n) with n={self.n}, got {theta_arr.shape}"
        )

        # batch-compute kinematics: returns arrays of shape (N, n)
        x_vec, y_vec, theta_abs_vec, anglesOfLinkEnds_vec, r_vec = self.DirectKinematics(theta_arr)

        # build column names
        column_names = ["x", "y", "theta_abs", "anglesOf_linksEnds", "r"]
        self.column_names_allLinks = [col+"_"+str(i) for col in column_names for i in range(self.n)]

        # concatenate along axis=1: results in shape (N, 5*n)
        data = np.concatenate((x_vec, y_vec, theta_abs_vec, anglesOfLinkEnds_vec, r_vec), axis=1)
        return pd.DataFrame(data, columns=self.column_names_allLinks)

    def set_function_exploration_region(self, dimension_limits):
        """
        Define sampling bounds for relative joint angles when generating datasets.

        Parameters
        ----------
        dimension_limits : tuple (inf, sup)
            Lower and upper bounds for each joint.
        """
        self.dimension_limit_inf, self.dimension_limit_sup = dimension_limits

    def generateDataset(self, n_samples):
        """
        Generate random samples of relative angles within defined bounds,
        compute kinematics, and store as a full DataFrame.

        Parameters
        ----------
        n_samples : int
            Number of samples.
        """
        if self.dimension_limit_inf is None or self.dimension_limit_sup is None:
            raise AssertionError("Define dimension limits first.")
        # Sample uniformly
        u = np.random.random_sample((n_samples, self.n))
        thetas = u * (self.dimension_limit_sup - self.dimension_limit_inf) + self.dimension_limit_inf
        self.x = pd.DataFrame(thetas, columns=[f"theta_rel_{i}" for i in range(self.n)])
        self.y = self.DirectKinematics_DataFrame(self.x.values)
        self.df_full = pd.concat([self.x, self.y], axis=1)

    def train_test_split_dataset(self, test_size):
        """
        Split the generated dataset into training and testing sets.

        Parameters
        ----------
        test_size : float
            Fraction of data to reserve for testing.
        """
        if not hasattr(self, 'df_full'):
            raise AssertionError("Generate dataset before splitting.")
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=test_size, random_state=self.random_state
        )

    def grafica_fromFinalEffector(self, x_n_desired, y_n_desired, theta_rel_fromIK,
                                  showDesiredPoint=True, showLinks=True, activateGrid=True):
        """
        Plot robot link positions and desired end-effector targets in 2D.

        Parameters
        ----------
        x_n_desired, y_n_desired : array-like
            Desired final end-effector coordinates.
        theta_rel_fromIK : array-like
            Joint angles from inverse kinematics.
        showDesiredPoint : bool
        showLinks : bool
        activateGrid : bool
        """
        x, y, _, _, _ = self.DirectKinematics(theta_rel_fromIK)
        # Include base origin in plotting
        x = np.insert(x, 0, 0, axis=1)
        y = np.insert(y, 0, 0, axis=1)
        fig = plt.figure()
        for xi, yi in zip(x, y):
            if showLinks:
                plt.plot(xi, yi)
            plt.scatter(xi, yi)
        if showDesiredPoint:
            for xd, yd in zip(x_n_desired, y_n_desired):
                plt.scatter(xd, yd, marker='*')
        plt.scatter(0, 0, s=100, c='k')
        plt.axis('equal')
        plt.grid(activateGrid)

    def saveDataset(self):
        """
        Save the generated dataset to CSV named by link lengths.

        Returns
        -------
        filename : str
            Path to the saved CSV file.
        """
        if not hasattr(self, 'df_full'):
            raise AssertionError("Generate dataset before saving.")
        name = "_".join(map(str, self.length_vec))
        filename = f"scara_robot_{name}.csv"
        self.df_full.to_csv(filename, index=False)
        return filename

    def computeJacobian(self, theta_rel_vec):
        """Computes and stores the Jacobian matrix J for an n-link planar SCARA robot, for the given vector of relative angles."""
        theta = np.array(theta_rel_vec)
        if theta.size != self.n:
            raise ValueError(f"Jacobian requires a {self.n}-vector of joint angles, got shape {theta.shape}.")

        # compute cumulative (absolute) angles for each link
        abs_angles = np.cumsum(theta)

        # initialize a 2 x n Jacobian matrix
        J = np.zeros((2, self.n))

        # for each joint j, accumulate contributions from link j through the last link
        for j in range(self.n):
            seg_lengths = self.length_vec[j:]     # lengths of links j, j+1, ..., n-1
            seg_angles = abs_angles[j:]           # their absolute angles

            # d x / d theta_j = - sum_{i=j..n-1} length_i * sin(angle_i)
            J[0, j] = -np.sum(seg_lengths * np.sin(seg_angles))
            # d y / d theta_j =   sum_{i=j..n-1} length_i * cos(angle_i)
            J[1, j] =  np.sum(seg_lengths * np.cos(seg_angles))

        self.jacobian = J
        return J

    def computePseudoInverseJacobian(self, theta_rel_vec):
        """
        Computes and stores the Moore-Penrose pseudoinverse of the Jacobian, for the given vector of relative angles.

        Parameters
        ----------
        theta_rel_vec : array-like
            Joint angles to update Jacobian if not already set.

        Returns
        -------
        J_pinv : np.ndarray
            Pseudoinverse of the Jacobian.
        """
        J = self.jacobian if self.jacobian is not None else self.computeJacobian(theta_rel_vec)
        J_pinv = np.linalg.pinv(J)
        self.pseudoInvOf_jacobian = J_pinv
        return J_pinv

from graphical_utils import grafica_fromCompleteVects, grafica_polar_fromCompleteVects

if __name__ == '__main__':
    env_robot = Robot_SCARA([20, 10])
    #env_robot = Robot_SCARA([10, 20, 30])

    x_vec, y_vec, theta_abs_vec, angle_vec, r_vec = env_robot.DirectKinematics([0, np.pi/2])
    #env_robot.DirectKinematics([0, np.pi/2, -np.pi/2])

    # Recall the description of the DK function:
    #    x_vec : np.ndarray, shape (n,)
    #        X coordinates of each link end.
    #    y_vec : np.ndarray, shape (n,)
    #        Y coordinates of each link end.
    #    theta_abs_vec : np.ndarray, shape (n,)
    #        Cumulative joint orientations (absolute angles of each joint).
    #    angle_vec : np.ndarray, shape (n,)
    #        Angle of the vector from the origin to each link tip (arctan2 of y_vec, x_vec).
    #    r_vec : np.ndarray, shape (n,)
    #        Radial distances from the base origin to each link end.

    grafica_fromCompleteVects(x_vec, y_vec, showLinks=True)

    grafica_polar_fromCompleteVects(r_vec, angle_vec)