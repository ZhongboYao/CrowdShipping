from simulation import Simulator
import configuration
import update as update
import configuration
import tools
import numpy as np
import properties
import visualization

"""
This file compares whether calculating the decay factor only according to the couriers arrival density feature works.
For now, the results exhibit the problem that the current decay method tends to assign parcels to the couriers 
with a large detour distance. The reason for this problem can be: 1) There are bugs in reading the detour values
in matching.py when calculating objective values. 2) The current decay_density method only takes the arrival population
into consideration, ignoring the couriers' destination coordinates, which can be useful in deciding the detour therefore
affecting the algorithm's performance.
"""

cost1 = 0
cost2 = 0

tools.destinations_generation()

# Run the simulation with decay method.
configuration.end_time = 8*60+30
simulator = Simulator(seed=40, save_log=0)
properties.timewindow_uniform_adjustment(simulator.tracker, [configuration.start_time, configuration.end_time-1])
simulator.solving(method='Hungarian', decay_enable=1)
simulator.result_process('simulations_log.npy', 'assignments_log.npy')
visualization.parcels_costs_analyse(list(simulator.tracker.parcels.values()), 'main/output/images/decay_enabled/')
cost1 = simulator.tracker._tot_paid

# Run the benchmark. (MyOpic without the decay method. The problem setup is exactly the same as the above one.)
configuration.end_time = 8*60+30
simulator = Simulator(seed=40, save_log=0)
properties.timewindow_uniform_adjustment(simulator.tracker, [configuration.start_time, configuration.end_time-1])
simulator.solving(method='Hungarian', decay_enable=0)
simulator.result_process('simulations_log.npy', 'assignments_log.npy')
visualization.parcels_costs_analyse(list(simulator.tracker.parcels.values()), 'main/output/images/decay_disabled/')
cost2 = simulator.tracker._tot_paid

print(cost1, cost2)
