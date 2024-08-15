from typing import Any, Dict, List
from fastapi import FastAPI, Body
import torch
import numpy as np
from .model import Autoencoder

app = FastAPI()

# Initialize the dictionary to hold the position and time data
# The structure is { "ID": {"latitude": [...], "longitude": [...], "time": [...], "mse": [...]} }
running_stats: Dict[str, Dict[str, List[float]]] = {}
WINDOW_SIZE = 100
MSE_WINDOW_SIZE = 50  # Number of MSE values to consider for running average
ANOMALY_THRESHOLD = 10  # Example threshold for mean MSE

# Create a new instance of the model
model = Autoencoder()

# Load the state dictionary
model.load_state_dict(torch.load('app/model.pth'))

# Set the model to evaluation mode
model.eval()

# Load the mean and std for normalization
mean_ = np.array([[14.99986662, 46.95354183]])  # Replace with the exact values
std_ = np.array([[2.99375461e-05, 1.97048753e-05]])  # Replace with the exact values

@app.post("/")
def read_root(payload: Any = Body(None)):
    global running_stats
    id_ = payload['ID']
    observed_latitude = payload['observed_latitude']
    observed_longitude = payload['observed_longitude']
    timestamp = payload['time']

    # Initialize the ID entry if it doesn't exist
    if id_ not in running_stats:
        running_stats[id_] = {
            "latitude": [],
            "longitude": [],
            "time": [],
            "mse": []  # Initialize MSE list
        }

    # Add the new data to the lists
    running_stats[id_]['latitude'].append(observed_latitude)
    running_stats[id_]['longitude'].append(observed_longitude)
    running_stats[id_]['time'].append(timestamp)

    # Ensure the lists maintain the fixed window size
    if len(running_stats[id_]['latitude']) > WINDOW_SIZE:
        running_stats[id_]['latitude'].pop(0)
        running_stats[id_]['longitude'].pop(0)
        running_stats[id_]['time'].pop(0)

    # Once we have enough data (i.e., WINDOW_SIZE), we can start processing
    if len(running_stats[id_]['latitude']) == WINDOW_SIZE:
        # Convert lists to a tensor
        data = np.array([running_stats[id_]['latitude'], running_stats[id_]['longitude']]).T
        data = (data - mean_) / std_  # Normalize using the provided mean and std
        data_tensor = torch.tensor(data, dtype=torch.float32).flatten().unsqueeze(0)

        # Predict using the model
        with torch.no_grad():
            output_tensor = model(data_tensor)
        
        # Calculate MSE
        mse = torch.mean((data_tensor - output_tensor) ** 2).item()

        # Add MSE to the running list for this ID
        running_stats[id_]['mse'].append(mse)

        # Maintain the MSE list window size
        if len(running_stats[id_]['mse']) > MSE_WINDOW_SIZE:
            running_stats[id_]['mse'].pop(0)

        # Calculate the mean of the MSEs for this ID
        mean_mse = np.mean(running_stats[id_]['mse'])

        # Check for anomaly based on the mean MSE
        if mean_mse > ANOMALY_THRESHOLD:
            print(f"Anomaly detected for ID {id_} with mean MSE: {mean_mse}, true_label {payload['state']}")

    return {"status": "success"}