import random 
import configuration
import instances
import properties as properties
import configuration
from instances import Tracker
from tqdm import tqdm
import tools
from instances import Parcel, Courier

class ParcelsUpdate:
    def parcels_login(tracker:Tracker, parcels_num:int=configuration.parcels_num, distribution:str="Random", parcel_clusters:list=None):
        """
        Generate and register parcels into the tracker.

        Parameters:
        tracker: Tracker instance
            A class used to store all current related information.
        parcels_num: int
            The number of parcels.
        distribution: str
            How the parcels are distributing.
            Can be 'Random', 'Normal' and 'Cluster'.
            If 'Cluster', parcel_clusters' value must be a positive integer.
        parcel_clusters: List[(center_x0, center_y0), (center_x1, center_y1) ...]
            A list of distribution centers if the parcels are distributing in clusters.
        """
        print('Parcels initialising...')

        with tqdm(total = parcels_num) as pbar: # Simple visualization of the process.
            for i in range(parcels_num):
                parcel = instances.Parcel(i)
                parcel.size = random.randint(1, configuration.parcels_max_size) # Generate sizes.

                # Generate coordinates for parcels.
                if distribution == "Random":
                    parcel.location = properties.coordinates_generation_random(configuration.instances_max_coordinate)
                elif distribution == "Normal":
                    parcel.location = properties.coordinates_generation_normal()
                elif distribution == "Cluster":
                    if parcel_clusters is None:
                        raise ValueError(f"Clusters are invalid.")
                    parcel.location = properties.coordinates_generation_cluster_random(parcel_clusters, configuration.parcel_cluster_dev)
                else:
                    raise ValueError(f"No ditribution method {distribution}")
                
                # Initialize time windows for the parcels.
                parcel.time_window = properties.timewindow_random_generation(
                    configuration.parcels_window_earliest_start_time, 
                    configuration.parcels_window_latest_end_time, 
                    configuration.parcels_minimum_window_length
                    )
                
                # Register the parcel in the tracker.
                tracker.parcels[parcel.index] = parcel

                pbar.update(1)
        
        tools.write_json(tracker.parcels, 'main/setup/', 'initialization_parcels.json')


    def parcels_time_update(tracker:Tracker, urgency_type:str="Linear", decay_enable=0) -> int:
        """
        Update all the parcels information.
        Turn them from standy to active, or from active to expired.
        Refresh the decay factor and urgency for the active parcels.

        Parameters: 
        tracker: Tracker instance
        urgency_type: str
            Can be 'Linear', 'Quadratic', 'neg_Quadratic', defined in properties.py.
        decay_enable: float
            The decay factor used to adjust a parcel's urgency.
            It is used in the dynamic problem scenario.

        Returns:
        expired_parcels_num: int
            The number of parcels that turned to expiration at the time step.
        """
        print('Updating Parcels\' Information ...')
        expired_parcels_num = 0

        with tqdm(total = len(tracker.parcels)) as pbar:
            for parcel in tracker.parcels.values(): 
                time_window = parcel.time_window
                
                # Turn frozen parcels into active state.
                if parcel.status == configuration.standby and tracker.t >= time_window[0]:
                    parcel.status = configuration.active
                    tracker.active_parcels[parcel.index] = parcel

                # Examine whether an active parcel now expires.
                if parcel.status == configuration.active and tracker.t > time_window[1]:
                    parcel.status = configuration.expired
                    parcel.carried_by = 'punished'
                    parcel.accepted_price = configuration.parameters['penalty']
                    tracker.active_parcels.pop(parcel.index) # Remove the expired ones.
                    expired_parcels_num += 1
                
                # Only update active parcels' information.
                if parcel.status == configuration.active:
                    # Calculate the decay if enabled.
                    if decay_enable:
                        properties.decay_factor_density(tracker, parcel, configuration.future_weight)

                    # Refresh the urgency.
                    if urgency_type == "Linear":
                        properties.urgency_linear(tracker, parcel)
                    elif urgency_type == "Quadratic":
                        properties.urgency_quadratic(tracker, parcel)
                    elif urgency_type == "neg_Quadratic":
                        properties.urgency_neg_quadratic(tracker, parcel)
                    else:
                        raise ValueError(f"Urgency type {urgency_type} is not supported!")

                pbar.update(1)

        return expired_parcels_num

    def parcel_accepted(tracker:Tracker, parcel:Parcel, courier:Courier, accepted_price:float):
        """
        Update the parcel's information if the assignment is accepted.

        Parameters:
        tracker, parcel, courier: Tracker, Parcel, Courier instances
        accepted_price: float
            The corresponding accepted price for this assignment.
        """
        parcel.carried_by = courier.index
        parcel.accepted_price = accepted_price
        parcel.accepted_detour = tracker.detour_memory[(courier.index, parcel.index)]

class CourierUpdate:
    def courier_login(tracker:Tracker, login_num:float, distribution:str="Random", courier_clusters:list=None, append_mode:int=0):
        """
        Simulating couriers login to the system according to the distribution pattern.
        Couriers' properties will be generated and stored in the Tracker.

        Parameters:
        tracker: Tracker instance
            Tracking the scenarios info.
        login_num: float
            The number of couriers logging in the system.
        distribution: str
            Couriers distribution pattern, including 'Random', 'Normal' and 'Cluster'.
        courier_clusters: list[(center_x0, center_y0), (center_x1, center_y1), ...]
            A list containing coordinates of cluster centers.
        seed: int
            Random seed.
        append_mode: int
            If append_mode == 1, then couriers are logging into the system every update 
            step instead of altogether registration at the beginning.
        """
        print('Registering couriers...')

        with tqdm(total = int(login_num)) as pbar:
            for i in range(int(login_num)):
                # Generate properties for the courier.
                index = len(tracker.couriers) # Assign an index for the new courier.
                courier = instances.Courier(index)
                courier.capacity = random.randint(1, configuration.couriers_max_capacity)

                if distribution == "Random":
                    courier.destination = properties.coordinates_generation_random(configuration.instances_max_coordinate)
                elif distribution == "Normal":
                    courier.destination = properties.coordinates_generation_normal()
                elif distribution == "Cluster":
                    if courier_clusters is None:
                        raise ValueError(f"Clusters must be defined before the generation.")
                    courier.destination = properties.coordinates_generation_cluster_random(courier_clusters, configuration.courier_cluster_dev)
                else:
                    raise ValueError(f"Distribution pattern {distribution} is not supported!")

                # Log couriers information to the scenario.
                tracker.couriers[index] = courier 
                tracker.idle_couriers[index] = courier

                pbar.update(1)

        tools.write_json(tracker.couriers, 'main/setup/', 'initialization_couriers.json', append_mode=append_mode)
        
    def couriers_time_update(tracker:Tracker) -> int:
        """
        This function tracks the time that couriers have stayed in the system, 
        and logs out the ones who have been idle for too much time in the system.

        Parameters:
        tracker: Tracker instance

        Returns:
        len(logout_couriers): int
            The amount of couriers that logged out.
        """
        print('Checking Left Couriers ...')
        logout_couriers = []

        with tqdm(total = len(tracker.idle_couriers)) as pbar:
            for courier in tracker.idle_couriers.values():
                courier.idle_time += 1
                if courier.idle_time > configuration.couriers_logout_patience:
                    logout_couriers.append(courier)
                    courier.status = configuration.left
                pbar.update(1)
        
        print('Loging Out Left Couriers ...')
        with tqdm(total = len(logout_couriers)) as pbar:
            for courier in logout_couriers:
                tracker.idle_couriers.pop(courier.index) # Pop the couriers logging out from the idle_couriers dict.
                pbar.update(1)
        
        return len(logout_couriers)

    def courier_accept(parcel:Parcel, courier:Courier, accepted_price:float, acceptance_ratio:float):
        """
        Update the courier's information if he decides to accept an assignment.

        Parameters:
        parcel, courier: Parcel and Courier instances.
        accepted_price, acceptance_ratio: float
            The corresponding acceptance price and ratio.
        """
        courier.carrying = parcel.index
        courier.accepted_offers = [parcel.index, accepted_price, acceptance_ratio]

    def courier_reject(parcel:Parcel, courier:Courier, rejected_price:float, acceptance_ratio:float):
        """
        Put the assignments the courier just rejects into the courier's rejected_offers attribute.
        The price is also recorded so that even if the courier A rejects the parcel B but the price has changed,
        courier A can still be inquired whether he would take the parcel B with the new price.

        Parameters:
        parcel, courier: Parcel and Courier instances
        rejected_price: float
            The offered price for the assignment, which the courier rejects.
        acceptance_ratio:
            The acceptance probability of the rejected assignment.
        """
        courier.rejected_offers.append([parcel.index, rejected_price, acceptance_ratio])
        