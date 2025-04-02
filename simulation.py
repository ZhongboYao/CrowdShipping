import update as update
import matching
from copy import deepcopy
import configuration
import visualization
import tools
import random
from instances import Tracker
import numpy as np

class Simulator:
    def __init__(
            self, 
            seed:int = 42,
            couriers_specified_num:int = None, 
            parcels_distribution:str = "Random",
            couriers_distribution:str = "Random", 
            urgency_type:str = "Linear",
            parcel_clusters_num:int = configuration.parcel_clusters_num,
            courier_clusters_num:int = configuration.courier_clusters_num,
            save_log:bool = 0
            ):

        """
        Initialize the attributes and set up the problem scenario.

        Parameters:
        seed : int
            Random state seed.
        couriers_specified_num : int
            Specified total couriers population at the problem beginning.
            If none, then the population prediction result is used as the population, 
            and only in this mode, can the couriers log into the system every update interval.
        parcels_distribution : 'Random' or 'Normal' or 'Cluster'
            How the parcels will distribute around the whole graph.
            If 'Random', then the parcel coordinations will be randomly generated.
            If 'Normal', then the parcel coordinations will distribution normally around a given center.
            If 'Cluster', then the parcel coordinations will distribution normally around multiple given centers.
        couriers_distribution : 'Random' or 'Normal' or 'Cluster'
            Similar as parcels_distribution.
        urgency_type : 'Linear' or 'Quadratic' or 'neg_Quadratic'
            It regulates how does a parcel's urgency change with time. 
            To find more details, see also properties.urgency_linear(), properties.urgency_quadratic(),
            properties.urgency_neg_quadratic().
        parcel_clusters_num: int
            The number of parcel clusters in parcels 'Cluster' distribution.
            Only useful if parcels_distribution == 'Cluster.
        courier_clusters_num: int
            The number of courier clusters in couriers 'Cluster' distribution.
            Only useful if couriers_distribution == 'Cluster.
        save_log: bool
            Whether the logs (containing information of trackers, assignments and all the other information 
            about the scenario) will be saved. Choosing not to save logs reduces the program's running time.
        """
        self.seed = seed
        self.couriers_specified_num = couriers_specified_num
        self.parcels_distribution = parcels_distribution
        self.couriers_distribution = couriers_distribution
        self.urgency_type = urgency_type
        self.parcel_clusters_num = parcel_clusters_num
        self.courier_clusters_num = courier_clusters_num
        self.save_log = save_log

        self.tracker = Tracker(configuration.start_time)
        self.parcel_clusters = None
        self.courier_clusters = None

        # Setup the problem scenario.
        self._problem_setup()

    def _problem_setup(self):
        """
        Initialise the problem:
        Arrival population information is stored.
        Parcels are initialized, and the information is stored in the system.
        Clusters for parcels and couriers are generated if they are 'Cluster' distributed.
        If couriers_specified_num is not None, couriers are registered together here.
        Logs can be saved if necessary.
        """
        random.seed(self.seed)

        # Generate clusters for the Cluster distribution.
        if self.parcels_distribution == "Cluster":
            if self.parcel_clusters_num is None or self.parcel_clusters_num <= 0:
                raise ValueError("Parcel cluster number must be a positive value.")
            else:
                self.parcel_clusters = tools.clusters_generation_random(self.parcel_clusters_num, configuration.parcel_clusters_coordinate_range)
                tools.save_as_npy(name='cluster_parcel.npy', folder='main/setup/', array=np.array(self.parcel_clusters))

        if self.couriers_distribution == "Cluster":
            if self.courier_clusters_num is None or self.courier_clusters_num <= 0:
                raise ValueError("Courier cluster number must be a positive value.")
            else:
                self.courier_clusters = tools.clusters_generation_random(self.courier_clusters_num, configuration.courier_clusters_coordinate_range)
                tools.save_as_npy(name='clusters_courier.npy', folder='main/setup/', array=np.array(self.courier_clusters))

        # Initialize parcels and register them in the system.
        update.ParcelsUpdate.parcels_login(self.tracker, configuration.parcels_num, self.parcels_distribution, self.parcel_clusters)

        # Register couriers-relevant information. 
        if self.couriers_specified_num is not None:
            # If it is not none, couriers are registered together.
            update.CourierUpdate.courier_login(self.tracker, self.couriers_specified_num, self.couriers_distribution, self.courier_clusters, 0)
        else:
            # Else, arrival population will be stored and used for couriers registration each update ste.
            self.population = tools.population_estimation(configuration.update_interval, configuration.station_name) * configuration.couriers_participation_rate
            self.tracker.population = self.population
                                           
        print('Problem scenario initialised!')

        # Save log?
        if self.save_log:
            configuration.simulations_log.append([deepcopy(self.tracker), self.tracker.t])
        
    
    def solving(self, method:str, decay_enable:int=0):
        """
        Execute the problem solving and simulations.

        Parameters:
        method: str, 'MILP' or 'Hungarian'
            This indicates which method will be selected for solving the problem
        courier_num: int
            The number of couriers.
        """

        while self.tracker.t <= configuration.end_time:
            optimization_check = 0
            t = tools.time_illustration(self.tracker.t)
            print(t.strftime('%H:%M'))

            # If there is a designated number of couriers, then login them according to the corresponding parameter.
            if self.couriers_specified_num is None: # Then couriers are logging in the system every update step.
                courier_num = self.population[self.tracker.t]
                update.CourierUpdate.courier_login(self.tracker, courier_num, self.couriers_distribution, self.courier_clusters, 1)

            # Update parcels and couriers information.
            parcels_expired_num = update.ParcelsUpdate.parcels_time_update(self.tracker, self.urgency_type, decay_enable) # In each timestep, the status of parcels will be updated.
            couriers_logout_num = update.CourierUpdate.couriers_time_update(self.tracker) # In each timestep, the status of couriers will be updated as well.

            # If it is an optimization step and there are couriers and parcels to be paired, a matching algorithm will be executed.
            if self.tracker.t % configuration.optimize_interval == 0 and len(self.tracker.idle_couriers) != 0 and len(self.tracker.active_parcels) != 0: 
                print(f'The current number of idle couriers: {len(self.tracker.idle_couriers)}')
                print(f'The current number of active parcels: {len(self.tracker.active_parcels)}')

                if method == 'MILP':
                    solver = matching.MILP(configuration.parameters, self.tracker)
                elif method == 'Hungarian':
                    solver = matching.Hungarian(configuration.parameters, self.tracker)
                else:
                    raise ValueError(f"Unsupported method '{method}'. Expected 'MILP' or 'Hungarian'")
                
                executor = matching.Execution(self.seed, self.tracker, solver)
                obj = executor.solve()
                assigned_parcels = executor.couriers_behave()
                optimization_check = 1

                if self.save_log:
                    configuration.assignments_log.append([deepcopy(executor), self.tracker.t])

            if self.save_log:
                configuration.simulations_log.append([deepcopy(self.tracker), self.tracker.t])

            # Result analysis.
            if optimization_check:
                self.tracker._obj.append(obj)
                self.tracker._assigned_couriers_num.append(len(assigned_parcels))
                self.tracker._assigned_parcels_num.append(len(assigned_parcels))
                for parcel in assigned_parcels:
                    self.tracker._tot_compensation += parcel.accepted_detour * parcel.accepted_price
            else:
                self.tracker._assigned_couriers_num.append(0)
                self.tracker._assigned_parcels_num.append(0)

            self.tracker._idle_couriers_num.append(len(self.tracker.idle_couriers))
            self.tracker._active_parcels_num.append(len(self.tracker.active_parcels))

            if self.couriers_specified_num is None:
                self.tracker._new_couriers_num.append(self.population[self.tracker.t])
            else:
                if len(self.tracker._new_couriers_num) == 0:
                    self.tracker._new_couriers_num.append(self.couriers_specified_num)
                else:
                    self.tracker._new_couriers_num.append(0)

            self.tracker._logout_couriers_num.append(couriers_logout_num)
            self.tracker._expired_parcels_num.append(parcels_expired_num)
            self.tracker._missed_penalty += parcels_expired_num * configuration.parameters['penalty']

            self.tracker.t += 1

    def result_process(self, simulation_file_name:str, assignments_file_name:str, visualize:int=0):
        """

        Parameters:
        simulation_file_name: str
            Directory for the folder saving simulation logs.
        assignments_file_name str
            Directory for the folder saving assignments logs.
        visualize: int
            If 1, assignments will be visualized.
        """
        self.tracker._cost_cal()
        if self.save_log:
            tools.save_output_log(configuration.simulations_log, 'main/output/logs/', simulation_file_name)
            tools.save_output_log(configuration.assignments_log, 'main/output/logs/', assignments_file_name)
            tools.save_output_script(self.tracker, 'main/output/logs/')
        if visualize:
            visualization.visualize_assignments(simulation_file_name)

