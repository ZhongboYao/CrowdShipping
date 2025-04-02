import configuration

class Courier:
    def __init__(self, index:int):
        """
        Parameters: 
        index: The index of this courier.
        """
        self.index = index
        self.capacity = None
        self.destination = None # (x, y) indicating a courier's original destination location.
        self.carrying = '-' # If this courier is assigned with a parcel, this attribute will be replaced by the index of the assigned parcel.
        self.accepted_offers = None # [accepted parcel index, the accepted price for the parcel]
        self.rejected_offers = [] # [[rejected parcel index, the rejected price], ...]
        self.idle_time = 0 # How much time (mins) has this courier been idle in the system.
        self.status = configuration.idle # Can be idle or left

    def to_dict(self) -> dict:
        """
        Transfer attributes to dictionary for storage in .json.

        Returns:
        A dictionary of attributes to be written into the .json file.
        """
        return{
            'index': self.index,
            'capacity': self.capacity,
            'destination': self.destination,
            'carrying': self.carrying,
            'accepted_offers': self.accepted_offers,
            'rejected_offers': self.rejected_offers,
            'idle_time': self.idle_time
        }

class Parcel:
    def __init__(self, index:int):
        """
        Parameters:
        index: int
            The index of the parcel
        """
        self.index = index
        self.size = None
        self.location = None # The parcel's destination, indicated by (x, y).
        self.carried_by = '-' # Will be replaced with the assigned courier's index, or 'punished' if expired.
        self.time_window = None # [start_time, end_time]
        self.status = configuration.standby # A parcel has three status: standby, active and (missed or expired).
        self.urgency = 0
        self.accepted_price = 0
        self.accepted_detour = 0
        # Decay is only used in Hungarian method, not in MILP method.
        self.decay = 1 # The variable used in the dynamic problem, making a parcel more or less urgent.

    def to_dict(self) -> dict:
        """
        Transfer attributes to dictionary for storage of the parcel's information in .json.

        Returns:
        A dictionary so that the attributes can be written into the .json file.
        """
        return{
            'index': self.index,
            'size': self.size,
            'location': self.location,
            'carried_by': self.carried_by,
            'time_window': self.time_window,
            'urgency': self.urgency,
            'accepted_price': self.accepted_price,
            'accepted_detour': self.accepted_detour,
            'decay': self.decay
        }
    
class Tracker:
    def __init__(self, start_time:int=configuration.start_time): 
        self.t = start_time # It records how many minutes are there from 00:00.
        # {0: parcel0, 1: parcel1, ...}, all the parcels in the system.
        self.parcels = {} 
        # Parcels that are valid for assignments, namely they are in their time window, not expired, haven't been assigned.
        # {index0, activeparcel_index0, ...}
        self.active_parcels = {} 
        # {0: courier0, 1: courier1, ...}, all the couriers that have appeared.
        # They will remain in this dict even if they get an assignment or they logged out.
        self.couriers = {} 
        # Couriers that are idle in the system, waiting for an assignment.
        # {index0, idlecourier_index0, ...}
        self.idle_couriers = {} 
        self.population = None # Couriers arrival info.
        # Invalid assignment pairs due to violations of constraints, such as the parcel's size exceeds the courier's capacity.
        # It is checked in matching phase to avoid attempts to make such assignments, reducing matching time.
        # {(courier_index0, parcel_index0):1, (courier_index1, parcel_index1):1, ...}
        self.not_match = {} 
        # Detour distances that have been calculated will be stored here to reduce running time.
        # {(courier_index0, parcel_index0): detour distance between them, ...}
        self.detour_memory = {}

        # The following variables are for plotting the graphs or experiments.
        self._tot_paid = 0
        self._missed_penalty = 0
        self._tot_compensation = 0

        self._obj = []
        self._idle_couriers_num = []
        self._new_couriers_num = []
        self._assigned_couriers_num = []
        self._logout_couriers_num = []

        self._active_parcels_num = []
        self._assigned_parcels_num = []
        self._expired_parcels_num = []

    def _cost_cal(self):
        """
        Calculates the cost results for illustration.
        """
        self._tot_paid = self._missed_penalty + self._tot_compensation
        saved = sum(self._assigned_couriers_num) * configuration.parameters['penalty'] - self._tot_compensation

        print(
            f"Couriers are compensated with Kr {self._tot_compensation} in total.\n"
            f"The expense for the system is Kr {self._tot_paid}.\n"
            f"The amount of money saved by hiring couriers is Kr {saved}."
        )