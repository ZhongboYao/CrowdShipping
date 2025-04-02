from simulation import Simulator
import configuration
import update as update
import configuration
import tools
import numpy as np
import visualization
import os
import properties

folder = f'main/output/images/experiments_plot/'
os.makedirs(os.path.dirname(folder), exist_ok=True)

def result_matching_check(instances_num_range=np.arange(1, 5, 1), random_state=45):
    """
    This function checks whether the objective value from MILP exactly matches with the 
    objective value from Hungarian algorithm.

    Parameters:
    instances_num_range: np.range
        The number of instances (both couriers and parcels) for each experiment.
    random_state: int
        Random seed.
    """

    cost_milp = []
    cost_hungarian = []
    
    configuration.end_time = configuration.start_time # Only optimize one step.
    configuration.parcels_minimum_window_length = 0 

    for num in instances_num_range:
        configuration.parcels_num = num
        milp_solve = Simulator(seed=random_state, couriers_specified_num=num, save_log=1)
        properties.timewindow_uniform_adjustment(milp_solve.tracker, [configuration.start_time-1, configuration.end_time+1])
        milp_solve.solving('Hungarian')
        milp_solve.result_process('simulations_log.npy', 'assignments_log.npy')
        cost_milp.append(milp_solve.tracker._obj)

        hungarian_solve = Simulator(seed=random_state, couriers_specified_num=num, save_log=1)
        properties.timewindow_uniform_adjustment(hungarian_solve.tracker, [configuration.start_time-1, configuration.end_time+1])
        hungarian_solve.solving('Hungarian')
        hungarian_solve.result_process('simulations_log.npy', 'assignments_log.npy')
        cost_hungarian.append(hungarian_solve.tracker._obj)

    visualization.plot_lines_compare(instances_num_range, [cost_milp, cost_hungarian], ["milp_obj", "hunagrian_obj"], f"{folder}result_match.jpg")

def ratios_plot():
    """
    An example showing how ratio plots work.
    """
    configuration.end_time = 22*60
    simulator = Simulator()
    simulator.solving('Hungarian')
    simulator.result_process('simulations_log.npy', 'assignments_log.npy')
    visualization.plot_ratio_couriers(simulator.tracker, f"{folder}CourierFeatures_ratio.jpg", configuration.start_time, configuration.end_time+1)
    visualization.plot_ratio_parcels(simulator.tracker, f"{folder}ParcelFeatures_ratio.jpg", configuration.start_time, configuration.end_time+1)

def simple_run():
    """
    Simply runs the simulation to check if it can function well.
    """
    configuration.end_time = 8*60+5
    simulator = Simulator(seed=40, couriers_specified_num=100, save_log=1)
    properties.timewindow_uniform_adjustment(simulator.tracker, [configuration.start_time, configuration.end_time-1])
    simulator.solving('Hungarian')
    simulator.result_process('simulations_log.npy', 'assignments_log.npy')
    visualization.parcels_costs_analyse(list(simulator.tracker.parcels.values()), folder)

result_matching_check()
ratios_plot()
simple_run()


