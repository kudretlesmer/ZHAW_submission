import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from pyproj import Proj, Transformer
import json
import socket
import json
import threading
from datetime import datetime


class Agent:
    def __init__(self, agent_id, parameters):
        # Initialize projection transformations for coordinate conversion
        self.utm_proj = Proj(proj='utm', zone=33, ellps='WGS84', preserve_units=False)
        self.wgs84_proj = Proj(proj='latlong', datum='WGS84')
        self.transformer = Transformer.from_proj(self.utm_proj, self.wgs84_proj)

        # Initialize agent properties
        self.id = agent_id  # Unique ID for each agent
        self.parameters = parameters
        self.state = self.initialize_state()
        self.position = np.array([500000.0, 5200000.0])  # Starting position (UTM coordinates)
        self.direction = 0  # Initial direction in degrees
        self.speed = 0  # Initial speed
        self.turning_angle = 0  # Initial turning angle
        self.neighbours = []  # Initialize an empty list for storing neighbors' IDs

    def initialize_state(self):
        # Initialize the state based on initial state probabilities
        states = self.parameters['states']
        state_probs = [self.parameters['initial_state_probabilities'][state] for state in states]
        initial_state = np.random.choice(states, p=state_probs)
        return initial_state

    def xy_to_wgs84(self, x, y):
        # Convert x/y (UTM coordinates) to lon/lat (WGS84 coordinates)
        lon, lat = self.transformer.transform(x, y)
        return lon, lat

    def get_ID(self):
        # Return the unique ID of the agent
        return str(self.id)

    def get_time(self):
        # Return the current time
        return str(datetime.now())

    def get_position(self):
        # Return the current position of the agent in WGS84 coordinates
        return self.xy_to_wgs84(self.position[0], self.position[1])

    def get_observed_position(self):
        # Return the observed position with added noise (simulating GPS error)
        x_error, y_error = np.random.multivariate_normal(self.parameters['Uxy'], self.parameters['Cxy'], 1)[0]
        return self.xy_to_wgs84(self.position[0] + x_error, self.position[1] + y_error)

    def get_direction(self):
        # Return the current direction of the agent
        return self.direction

    def get_state(self):
        # Return the current state of the agent
        return self.state

    def get_speed(self):
        # Return the current speed of the agent
        return self.speed

    def get_neighbours_speed(self, states_df):
        # Calculate the average speed of the agent's neighbors
        speed = sum(states_df[states_df['ID'] == neighbour]['speed'].iloc[0] for neighbour in self.neighbours)
        return speed / len(self.neighbours) if self.neighbours else 0

    def get_neighbours_direction(self, states_df):
        # Calculate the average direction of the agent's neighbors
        direction = sum(states_df[states_df['ID'] == neighbour]['direction'].iloc[0] for neighbour in self.neighbours)
        return direction / len(self.neighbours) if self.neighbours else 0

    def update_transition_probs(self):
        # Update the transition probabilities based on the current state
        self.transition_probs = self.parameters['state_transition_matrix'][self.state]

    def update_state(self, states_df):
        # Check if any of the neighbors are in the "Fleeing" state
        fleeing_neighbors = states_df[states_df['ID'].isin(self.neighbours) & (states_df['state'] == 'Fleeing')]
        
        if not fleeing_neighbors.empty:
            # If at least one neighbor is fleeing, the agent also changes to "Fleeing" state
            self.state = 'Fleeing'
        else:
            # Otherwise, update the state based on transition probabilities
            self.state = np.random.choice(self.parameters['states'], p=self.transition_probs)


    def update_state_params(self):
        # Update the parameters associated with the current state
        self.state_params = self.parameters['state_parameters'][self.state]

    def update_speed(self, states_df):
        # Update the speed of the agent considering neighbors' speed and randomness
        neighbours_speed = self.get_neighbours_speed(states_df)
        min_speed, max_speed = self.state_params['speed']
        self.speed = (neighbours_speed * self.parameters['weight_factor'] +
                      (1 - self.parameters['weight_factor']) * np.random.uniform(min_speed, max_speed))

    def update_direction(self, states_df):
        # Update the direction of the agent considering neighbors' direction and randomness
        neighbours_direction = self.get_neighbours_direction(states_df)
        if self.state in ['Grazing', 'Resting']:
            # Use a uniform distribution for turning angles
            min_angle, max_angle = self.state_params['turning_angle']
            self.turning_angle = (neighbours_direction * self.parameters['weight_factor'] +
                                  (1 - self.parameters['weight_factor']) * np.random.uniform(min_angle, max_angle))
        elif self.state in ['foraging', 'Fleeing']:
            # Use a Gaussian distribution for turning angles
            mean_angle, std_dev = self.state_params['turning_angle']
            self.turning_angle = (neighbours_direction * self.parameters['weight_factor'] +
                                  (1 - self.parameters['weight_factor']) * np.random.normal(mean_angle, std_dev))
        # Ensure the direction is within 0-360 degrees
        self.direction = (self.direction + self.turning_angle) % 360

    def update_parameters(self, states_df):
        # Update all parameters after each step
        self.update_transition_probs()
        self.update_state(states_df)
        self.update_state_params()
        self.update_speed(states_df)
        self.update_direction(states_df)

    def step_forward(self, states_df):
        # Move the agent forward by updating position based on speed and direction
        angle_rad = np.deg2rad(self.direction)
        delta_x = self.speed * np.cos(angle_rad) / self.parameters['fs']
        delta_y = self.speed * np.sin(angle_rad) / self.parameters['fs']
        self.position += np.array([delta_x, delta_y])
        # Update neighbors after moving
        self.neighbours = self.get_n_closest_neighbours(states_df, self.parameters['neighbours'])
        # Update all other parameters based on new position and neighbors
        self.update_parameters(states_df)

    def get_update(self):
        # Prepare and return a dictionary of the agent's current state and position
        out_dict = {
            'ID': self.get_ID(),
            'real_latitude': self.get_position()[0],
            'real_longitude': self.get_position()[1],
            'observed_latitude': self.get_observed_position()[0],
            'observed_longitude': self.get_observed_position()[1],
            'speed': self.speed,
            'direction': self.direction,
            'state': self.state,
            'time': self.get_time()
        }
        return out_dict

    def get_n_closest_neighbours(self, updates, N):
        # Calculate distances to all other agents and return the IDs of the N closest ones
        agent_position = updates[updates['ID'] == self.get_ID()][['real_latitude', 'real_longitude']].values[0]
        lat_diff = updates['real_latitude'].values - agent_position[0]
        lon_diff = updates['real_longitude'].values - agent_position[1]
        distances = lat_diff**2 + lon_diff**2  # Squared distances for comparison
        updates['distance'] = distances
        closest_agents = updates[updates['ID'] != self.get_ID()].nsmallest(N, 'distance')
        del updates['distance']  # Clean up the DataFrame
        return closest_agents['ID'].tolist()