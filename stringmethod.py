import numpy as np
from scipy.interpolate import interp1d
import torch

class StringMethod:
    def __init__(self, initial_string, force_function, max_iterations=3000, dt=0.01):
        """
        Initialize the String Method for finding minimum energy paths.
        
        Args:
            initial_string (np.ndarray): Initial string coordinates
            force_function: External function that calculates forces
            spring_constant (float): Spring constant for string forces
        """
        self.initial_string = initial_string
        self.force_function = force_function

        # Method parameters
        self.max_iterations = max_iterations
        self.tolerance = 1e-4
        self.dt = dt

    def reparametrize_string(self, string):
        """Reparametrize the string to maintain equal spacing between points.

        Args:
            string (np.ndarray): Array of shape (n_points, n_atoms, 3)
        """
        # Flatten the atomic coordinates for distance calculation
        string_flat = string.reshape(len(string), -1)

        # Calculate cumulative distances
        cumulative_dist = np.concatenate(([0], np.cumsum(np.linalg.norm(np.diff(string_flat, axis=0), axis=1))))
        total_dist = cumulative_dist[-1]
        new_cumulative_dist = np.linspace(0, total_dist, len(string))

        # Interpolate each atomic coordinate separately
        interpolator = interp1d(cumulative_dist, string_flat, axis=0, kind='linear')
        new_string_flat = interpolator(new_cumulative_dist)

        # Reshape back to original dimensions
        new_string = new_string_flat.reshape(len(string), *string.shape[1:])

        return new_string

    def equilibrate(self):
        """Run the string method until convergence."""
        string = self.initial_string.copy()

        for iteration in range(self.max_iterations):
            # Calculate external and spring forces
            potential_force = self.force_function(string)

            # Adaptive time stepping
            #force_mag = np.linalg.norm(total_force, axis=1)
            #dt = min(self.dt_max, 0.1 / np.max(force_mag)) if np.max(force_mag) > 0 else self.dt_max

            string_new = string.copy()
            string_new += self.dt * potential_force

            # Reparametrize the string (excluding endpoints)
            string_new[1:-1] = self.reparametrize_string(string_new)[1:-1]

            # Check convergence
            displacement = np.max(np.linalg.norm(string_new[1:-1] - string[1:-1], axis=1))
            if displacement < self.tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break

            string = string_new.copy()

        return string


