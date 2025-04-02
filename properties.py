import random
import configuration
import numpy as np
import math
import matching
from copy import deepcopy
from instances import Tracker, Parcel

LARGE_DECAY = 5

def coordinates_generation_random(range:float=configuration.instances_max_coordinate) -> tuple:
    """
    Randomly generate a coordinate with both x and y are within the range.

    Parameters:
    range: float
        The maximum absolute value of both x and y.

    Returns:
    coordinate: tuple(x, y)
        A coordinate (x, y) in tuple.
    """
    x = random.uniform(-range, range)
    y = random.uniform(-range, range)
    return (x, y)

def coordinates_generation_normal(mean_x:float=0, mean_y:float=0, std_dev:float=configuration.instances_max_coordinate) -> tuple:
    """
    Given the center (mean_x, mean_y), this function generates a coordinate distributing normally around the center with the variance = std_dev.

    Parameters:
    mean_x, mean_y: float
        The center around which parcels are distributing.
    std_dev: float
        The max deviation distance from the center.
        It applies to both x and y.

    Returns:
    (x, y): Tuple(float, float)
        The generated coordinate.
    """
    x = np.random.normal(mean_x, std_dev)
    y = np.random.normal(mean_y, std_dev)
    return (x, y)

def coordinates_generation_cluster_random(clusters_positions:list=[(0, 0)], std_dev:float=configuration.courier_cluster_dev) -> tuple:
    """
    Give clusters' coordinates, this function generates a coordinate distributing normally around the cluster centers.

    Parameters:
    clusters_positions: List[(center_x0, center_y0), (center_x1, center_y1), ...]
        A list containing coordinates of the clusters.
    std_dev: float
        The largest deviation distance from the center.

    Returns:
    (x, y): Tuple(float, float)
        The generated coordinate.
    """
    # It firstly randomly selects a cluster, around which coordinates are distributing.
    cluster_index = random.randint(0, len(clusters_positions)-1)
    cluster_location = clusters_positions[cluster_index]

    # Then it generates a coordinate normally distributing around the selected cluster center.
    x, y = coordinates_generation_normal(mean_x=cluster_location[0], mean_y=cluster_location[1], std_dev=std_dev)
    return (x, y)

def urgency_linear(tracker:Tracker, parcel:Parcel):
    """
    Use a linear function to calculate a parcel's urgency.
    """
    time_window = parcel.time_window
    parcel.urgency = (tracker.t - time_window[0])/(time_window[1] - time_window[0])

def urgency_quadratic(tracker:Tracker, parcel:Parcel):
    """
    Use a quadratic function to calculate a parcel's urgency.
    """
    time_window = parcel.time_window
    parcel.urgency = ((tracker.t - time_window[0])/(time_window[1] - time_window[0])) ** 2

def urgency_neg_quadratic(tracker, parcel):
    """
    Use a quadrac function, simliar to the urgency_quadratic but reversed against x-axis and moved 
    up a bit, to calculate the urgency.
    """
    time_window = parcel.time_window
    parcel.urgency = 1 - ((tracker.t - time_window[1])/(time_window[1] - time_window[0])) ** 2

def urgency_given_time(t:float, parcel:Parcel):
    """
    This function calculates the urgency of a parcel in the future.

    Parameters:
    t: float
        Time in the future.
    parcel: Parcel
        Whose urgency needs to be updated.
    """
    time_window = parcel.time_window
    parcel.urgency = (t - time_window[0])/(time_window[1] - time_window[0])

def decay_factor_comparison(tracker:Tracker, parcel:Parcel, destination_list:list):
    """
    This method is abandoned due to previous discussions. This method adjusts the decay factor according 
    to a future score. In this method, the optimizer first solves the problem as a myopic version, getting 
    a pre-assigned courier. Then, it calculates the objective values of the predicted future couriers, and 
    compares the future ones with the pre-assigned one. If the objective score is lower than the pre-
    assigned one's, then that courier is more promising. All the promising couriers will contribute to the 
    future score. The function calculating the future score is not carefully tuned yet. But this function 
    for now cannot make a parcel more urgent.

    Parameters:
    tracker: Tracker instance
        Current tracker
    parcel: Parcel instance
        The parcel whose decay factore is being adjusted.
    destination_list: List[Courier0's destination, Courier1's destination, ...]
        It contains the coordinates of couriers' destinations.
        Coureris in the list should be sorted according to their arrival time.
    """
    predict_length = configuration.predict_length
    login_nums = tracker.population[tracker.t:tracker.t+predict_length+1]
    previous_login_num = int(sum(tracker.population[:tracker.t]))
    decay = 0
    N = 0
    for t in range(predict_length):
        login_num = int(login_nums[t])
        destinations = destination_list[previous_login_num:previous_login_num+login_num]
        previous_login_num += login_num
        parcel_loc = parcel.location
        for destination in destinations:
            distance = ((destination[0] - parcel_loc[0]) ** 2 + (destination[1] - parcel_loc[1]) ** 2) ** (1/2)
            if distance <= configuration.predict_threshold:
                time_left = parcel.time_window[1] - tracker.t - t
                shadow_parcel = deepcopy(parcel)
                urgency_given_time(tracker.t+t, shadow_parcel)
                _, new_weight = matching.find_best_weight(shadow_parcel, distance)
                if time_left - 1 > 0 and parcel.best_weight - new_weight/5 >= 1:
                    decay_temp = 0.3 / (parcel.best_weight - new_weight/5) + 0.7
                    decay += decay_temp
                    N += 1
    if N != 0:
        decay_final = decay/N
        parcel.decay = decay_final
    else:
        parcel.decay = 1

def decay_factor_density(tracker:Tracker, parcel:Parcel, future_weight:float=configuration.future_weight):
    """
    This function dynamically adjusts parcels' urgency (whether to be assigned) by 
    calculating the decay factor according to the couriers' arrival density.
    The decay factor will be multiplied with the urgency to make it more or less urgent
    in other functions.

    This is an experimental strategy for the dynamic problem version, and should be 
    improved further. The performance is usually good but there are exceptions.
    The best values for hyperparameters like predict_length, future_weight 
    and the other similar ones are correlated the number of parcels and couriers.
    Therefore it's better to conduct experiments to find the best settings or to make 
    it smarter (like the futher future can have a lowe weight compared with a near future).
    A problem is that this function only considers arrival population. But taking the 
    population of all the idle couriers at that future time step into consideration may
    be more reasonable?

    Parameters:
    tracker: Tracker instance
    parcel: Parcel instance
    future_weight: float
        A hyperparameter weighing the importance of futher scores.
    """
    future_score = 0
    predict_length = configuration.predict_length

    # Get the arrival population information within the prediction scope.
    end_time = tracker.t + predict_length + 1
    if end_time > parcel.time_window[1] + 1: # Couriers arriving after a parcels has expired are not considered.
        end_time = parcel.time_window[1] + 1
    login_nums = tracker.population[tracker.t+1:end_time]

    time_left_current = parcel.time_window[1] - tracker.t # Time left before the parcel's expiration time.
    current_score = len(tracker.idle_couriers) * time_left_current

    for i in range(len(login_nums)):
        login_num = int(login_nums[i])
        time = tracker.t + 1 + i
        time_left_future = parcel.time_window[1] - time # In that future time step, how many minutes a parcel will have before expiration.
        future_score += login_num * time_left_future * future_weight

    # If currently there are no available couriers for this parcel.
    if current_score == 0:
        decay = 1 
    # If in the future there will be no couriers logging in the system.
    elif future_score == 0:
        decay = LARGE_DECAY
    else:
        decay = current_score/future_score
    parcel.decay = decay

def timewindow_random_generation(
        start_boundary:int = configuration.parcels_window_earliest_start_time, 
        end_boundary:int = configuration.parcels_window_latest_end_time,
        minimum_window_length:int = configuration.parcels_minimum_window_length
        ) -> list:
    """
    Randomly generate a time window within in the given range defined by [start_boundary, end_boundary].
    The window's length will at least be equal to minimum_window_length.

    Parameters:
    start_boundary & end_boundary: int
        Boundaries for the generated time window
    minimum_window_length: int
        The minimum length of the time window.

    Returns:
    [start_time, end_time]: list[int, int]
        A time window.
    """
    start_time = random.randint(start_boundary, end_boundary - minimum_window_length)
    end_time = random.randint(start_time + minimum_window_length, end_boundary)
    return [start_time, end_time]

def timewindow_uniform_adjustment(tracker:Tracker, specified_time_window:list):
    """
    Regenerates time windows as the specified time window for all the parcels contained in the tracker.
    
    Parameters:
    tracker: Tracker instance
    specified_time_window: list
        The specified time window for parcels.
    """
    for parcel in tracker.parcels.values():
        parcel.time_window = specified_time_window

def timewindow_random_adjustment(tracker:Tracker, start_time:float, end_time:float):
    """
    Randomly regenerates time windows within the given range for all the parcels contained in the tracker.
    
    Parameters:
    tracker: Tracker instance
    start_time, end_time: float
        The range of time windows.
    """
    for parcel in tracker.parcels.values():
        parcel.time_window = timewindow_random_generation(start_time, end_time-1)

def individual_timewindow_generation_specified(start_time_mean:float, latest_time:float, uniform_length:float, variation:float) -> list:
    """
    This function generates a time window according to a distribution satisfying the given features.

    Parameters:
    start_time_mean: float
        The mean value of start time.
    latest_time: float
        The lastest time at which a time window can end.
    uniform_length: float
        The length of the time window.
    variation: float
        Variation of start time.

    Returns:
    A list indicating the time window.
    """
    random.seed(42)
    start = np.random.normal(start_time_mean, variation)
    end = start + uniform_length
    if end > latest_time:
        end = latest_time
    return [start, end]

def timewindow_scenarios_adjustment(tracker:Tracker, start_type:str, length_type:str, start_time:float, end_time:float):
    """
    The function reshapes all the parcels' time windows so that they follow some features. There are 4 types of time
    window settings, namely (early or late) combined with (short or long). Early or late means whether the start
    time of parcels is early or late. Short or long means whether the time window's length is short or long.

    Parameters:
    tracker: Tracker instance
    start_type: str
        It can be either 'early' or 'late'
    length_type: str
        It can be either 'short' or 'long'
    start_time, end_time: float
        The start and end time of the simulation, specified in the configuration.        
    """
    life_span = end_time - start_time

    if start_type == 'early':
        start_mean = start_time + 0.05 * life_span
    if start_type == 'late':
        start_mean = end_time - 0.4 * life_span
    if length_type == 'short':
        length = 0.2 * life_span
    if length_type == 'long':
        length = 0.5 * life_span
        
    for parcel in tracker.parcels.values():
        parcel.time_window = individual_timewindow_generation_specified(start_mean, end_time-1, length, 0.1 * life_span)
        