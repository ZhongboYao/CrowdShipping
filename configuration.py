global simulation_log
global assignments_log

start_time = 8*60 # Simulation start time point in minutes.
end_time = 22*60 # Simulation end time point in minutes.

station_name = 'Bergshamra' # Name of the station to extract arrival population information.
predict_length = 20 # How many minutes of future to consider in dynamic scenarios.
predict_threshold = 5 # (Only effective in the comparison method of calcuating the decay) Parcels' 

future_weight = 0.1 # Weight of future when calculating the decay factor.

# Track all the changes of parcels, couriers acception/rejection.
# In the form of [(Tracker, time), ...]
simulations_log = [] 
# Track all the assignments.
# In the forma of [(Executor, time), ...]
assignments_log =[]

update_interval = 1 # How many MINUTES are there in an interval of updating couriers' and parcels' info.
optimize_interval = 1 # How many MINUTES are there in an interval of optimizing the assignments.

parcels_num = 100 # The number of parcels.
parcels_max_size = 10 # The maximum size of parcels.
parcels_window_earliest_start_time = start_time # The earliest time window start time.
parcels_window_latest_end_time = end_time # The latest time window stop time.
parcels_minimum_window_length = 2 # The minimum length of time windows.

couriers_max_capacity = 10 # The maximum capcaity of couriers.
couriers_logout_patience = 100 # After how many minutes a courier will logout.
couriers_participation_rate = 0.0005 # The ratio of the public serving as couriers.

instances_max_coordinate = 10 # The max coordinates of parcels and couriers' destinations, for both x and y.
assignments_detour_threshold = 10000 # Detour beyond this threshold will not be considered for an assignment.

heuristic_price_levels_num = 200 # The number of price levels seached through in the heuristic methods.

# Configurations for colors in plotting for each status.
standby = (0.322, 0.322, 0.322)
active = (0.204, 0.89, 0.263)
assigned = (0.902, 0.208, 0.208)
expired = (0.902, 0.208, 0.208)
idle = (0.204, 0.89, 0.263)
left = (0.902, 0.208, 0.208)

parameters = {
    'price_levels_num': 200, # The number of price levels searched by the optimiser.
    'a': 0.192181 * 100, # a, b, w, epsilin are parameters in the utility function.
    'b': 0.031404 * 100,
    'w': 0.139685 * 100,
    'epsilon': -120,
    'penalty': 100, # Penalty if a parcel expired and hasn't been delivered.
    'U': 100 # Compensation upperbound.
}

# Number of clusters.
courier_clusters_num = 3
parcel_clusters_num = 3

# The largest distance between the farest sample to the center (equals to the variance in normal distribution).
courier_cluster_dev = 1
parcel_cluster_dev = 1

# Cluster coordinates will be generated within this range, for both x and y.
parcel_clusters_coordinate_range = 10 
courier_clusters_coordinate_range = 10