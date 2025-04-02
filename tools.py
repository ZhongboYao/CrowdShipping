import json
from typing import Tuple
import configuration
import numpy as np
import pickle
import numpy as np
from datetime import time
import properties
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import os
from instances import Tracker, Parcel


def distance_cal(coordinate1:tuple, coordinate2:tuple) -> float:
    """
    Calculates the Euclidean distance between two corrdinates.

    Parameters:
    coordinate1 & 2: Tuple(float, float)
        The two coordinates.

    Returns:
    round(distance, 2): float
        The Euclidean distances, rounded to 2 digits in the fraction part.
    """
    distance = ((coordinate1[0] - coordinate2[0]) ** 2 + (coordinate1[1] - coordinate2[1]) ** 2) ** (1/2)
    return round(distance, 2)

def clusters_generation_random(clusters_num:int=3, coordinate_range:float=configuration.instances_max_coordinate) -> list:
    """
    Given the wanted number of clusters, the function returns a list containing coordinates of the clusters that generated randomly.

    Parameters:
    seed: int
        Random seed.
    clusters_num: int
        The number of clusters.
        Should be an integer value >= 1.
    coordinate_range: float
        The absolute max value of both x and y in the clusters' coordinates.

    Returns:
    clusters_positions: [(x0, y0), (x1, y1), ...]
        A list of randomly generated cluster coordinates.
    """
    clusters_positions = []
    for _ in range(clusters_num):
        clusters_positions.append(properties.coordinates_generation_random(coordinate_range))
    return clusters_positions

def population_estimation(interval:int=1, station_name:str='Bergshamra') -> list:
    """
    Use the model regressing the arrival population each hour at a station to estimate the 
    arrival population each several minutes at the station. 

    Parameters:
    interval: int
        The number of MINUTES each interval contains for estimate the arrival population.
    station_name: str
        The name of the station to estimate.
    
    Returns:
    estimated_populations: list[population1, p2, p3 ...]
        A list containing arrival population each time step.
    """
    model_path = f'main/input/arrivals/Spline_Model_{station_name}.pkl'
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    time_steps = np.linspace(0, 24, int(24 * 60 / interval)) # Calculate how many time steps are there given the interval.
    estimated_populations = model(time_steps)
    print(f'Population info for station {station_name} is retrieved!')

    return estimated_populations

def write_json(hash:dict, folder:str, filename:str, append_mode:int=0):
    """
    Write the information in the hashtable into a json file.

    Parameters:
    hash: dict
        The dict to be stored
    folder: str
        Folder directory to store the dict.
    filename: str
        File name.
    append_mode: int
        If 1, the funciton adds content into the file rather than replace the original one.
    """
    os.makedirs(os.path.dirname(folder), exist_ok=True)
    instances_list = [hash[instance].to_dict() for instance in hash]
    instances_json = json.dumps(instances_list, indent=4)
    if not append_mode:
        with open(f'{folder}/{filename}', "w") as file:
            file.write(instances_json)
    else:
        with open('filename.txt', 'a') as file:
            file.write(instances_json)

def time_illustration(time_:float) -> time:
    """
    Given a value of total minutes, this function returns a standard time instance.

    Parameters:
    time_: float
        The total number of values from 00:00.
    
    Returns: 
    t: Time instance
    """
    hours = time_ // 60
    minutes = time_ % 60
    t = time(hour=hours, minute=minutes)
    return t

def destinations_generation(station_name=configuration.station_name, loc='main/input/specified_coordinates'):
    """
    Pre-define destination coordinates of the arriving couriers.

    Parameters:
    station_name: str
        The name of the station whose arrival population will be used.
    loc: str
        The folder to save the destination coordinations files.
    """
    specified_coordinates_total = []
    login_nums = population_estimation(1, station_name) * configuration.couriers_participation_rate

    for t in range(len(login_nums)):
        for _ in range(int(login_nums[t])):
            coordinate = properties.coordinates_generation_random()
            specified_coordinates_total.append(coordinate)

    specified_coordinates_total_np = np.array(specified_coordinates_total, dtype=object)
    np.save(f"{loc}.npy", specified_coordinates_total_np)
    np.savetxt(f"{loc}.csv", specified_coordinates_total_np, delimiter=',')

def save_as_npy(name:str, folder:str, array:np.array):
    """
    Save a numpy array as a .npy file in the disgnated folder.

    Parameters:
    name: str
        The name of the file containing the array.
    folder: str
        Folder directory.
    array: numpy array
        A numpy array that is going to be saved.
    """
    os.makedirs(os.path.dirname(folder), exist_ok=True)
    np.save(f'{folder}/{name}', array)

def save_output_script(tracker:Tracker, folder:str):
    """
    Save the all the scenario recorded by Tracker into .json files.

    Parameters:
    tracker: Tracker instance
    folder: str
        Folder directory.
    """
    os.makedirs(os.path.dirname(folder), exist_ok=True)
    write_json(tracker.parcels, 'main/output/logs/', 'result_parcels.json')
    write_json(tracker.couriers, 'main/output/logs/', 'result_couriers.json')

def save_output_log(simulation_log:list, folder:str, filename:str):
    """
    Save the results into a .npy file for visulization.

    Parameters:
    simulation_log: List[(tracker1, t1), (track2, t2), ...]
        The simulation log, a list containing all the trackers.
    folder: str
        Folder directory to save the file.
    filename: str
        The name of the file containing the log.
    """
    os.makedirs(os.path.dirname(folder), exist_ok=True)
    simulation_log_np = np.array(simulation_log)
    np.save(f'{folder}/{filename}', simulation_log_np)