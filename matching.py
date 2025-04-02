import acception
import tools
import configuration
from instances import Courier, Tracker
from tqdm import tqdm
from instances import Parcel
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import math
from scipy.optimize import linear_sum_assignment
from scipy.optimize import minimize
import update
import keywords

LARGE = 10000000 # A large number in the cost matrix prevent assignments.

def acceptance_ratio_calculation(price:float, detour:float, parcel_size:int) -> float:
    """
    Calculates the acceptance ratio using the binary logistic model.

    Parameters:
    price: float
        Compensation rate.
    detour: float
        Detour distance(Km) between a parcel and a courier.
    parcel_size: float
        The size of the parcel.

    Returns:
    acc: float
        The probability of accepting the assignment.
    """
    utility = (price * configuration.parameters['b'] - configuration.parameters['a']) * detour - configuration.parameters['w'] * parcel_size

    # To avoid overflow and underflow.
    if utility <= -708:
        utility = -708
    elif utility >= 708:
        utility = 708

    acc = 1/(1 + math.exp(-1/50 * utility))
    if acc <= 1e-3 or price == 0: # Ignore the acceptance that is too small (<1e-3).
        acc = 0

    return acc

def weight_calculation(price:float, detour:float, urgency:float, parcel_size:int, decay:int) -> float:
    """
    Calculates the objective value (weight) of teh assignment between a parcel and a courier.

    Parameters:
    price: float
        Compensation rate.
    detour: float
        Detour distance(Km) between a parcel and a courier.
    urgency: fliat
        Urgency of a parcel
    parcel_size: float
        The size of the parcel.
    deay: float
        The decay factor used in the dynamic scenario, adjusting the parcel's urgency.

    Returns:
    obj: float
        The objective value of a (courier, parcel) pair.
    """
    acc = acceptance_ratio_calculation(price, detour, parcel_size)
    obj = acc * (price * detour - urgency * configuration.parameters['penalty'] * decay)
    return obj

def find_best_weight(parcel:Parcel, detour:float, initial_price_guess=configuration.parameters['U']/5) -> tuple:
    """
    Find the best price and weight between a courier and a parcel for minimizing the objective.
    The decay factor is also included here. But if it isn't enabled in the simulator, it will not bring
    effect as it is set to 1 as default.

    Parameters:
    parcel: Parcel instance
    detour: float
        The detour distance between this parcel and the corresponding courier.
    initial_price_guress: float
        The initial start value for finding the best value.

    Returns:
    optimal_price: float
        The best price that minimizes the objective.
    optimal_weight: float
    """
    result = minimize(weight_calculation, initial_price_guess, args=(detour, parcel.urgency, parcel.size, parcel.decay), method="COBYLA", bounds=[(0, configuration.parameters['U'])])
    optimal_price = result.x[0]
    optimal_weight = result.fun
    return optimal_price, optimal_weight

def get_detour(tracker:Tracker, detour_entry:float, courier:Courier, parcel:Parcel) -> float:
    """
    Calculate the unrevealed detour between the given Courier and Parcel.
    If detour_entry == 0, then it hasn't been revealed.
    If it is memorized in detour_memory in tracker, which stores detours that have been calculated, then fetch it.
    Otherwise, calculate it.
    If detour_entry != 0, it means the value is already meaningful. Simply skip.

    Parameters:
    tracker: Tracker instance
    detour_entry: float
        It's from detour[i, j], indicating the detour distance between the (courier, parcel) pair.
    courier, parcel: Courier and Parcel instances
        The courier and parcel related to this detour entry.

    Returns:
    detour_entry: float
        The updated detour entry.
    """
    if detour_entry == 0:
        if (courier.index, parcel.index) in tracker.detour_memory:
            detour_entry = tracker.detour_memory[(courier.index, parcel.index)]
        else:
            detour_entry = tools.distance_cal(courier.destination, parcel.location) 
            tracker.detour_memory[(courier.index, parcel.index)] = detour_entry
    return detour_entry

def capacity_judgement(courier:Courier, parcel:Parcel) -> bool:
    """
    Check whether the courier's capacity is larger than the parcel's size.

    Parameters:
    courier, parcel: Courier and Parcel instances
        The pair to be checked.

    Returns:
    A boolean result indicating whether the capacity is sufficient.
    Return 1 is capacity is large enough.
    """
    if courier.capacity >= parcel.size:
        return True
    else:
        return False

# Current solver will still assign a courier with a rejected proposal.
class MILP:
    def __init__(self, parameters:dict, tracker:Tracker):
        """
        Parameters:
        parameters: dict
            Hyperparameters used in the MILP solver.
        tracker: Tracker instance
        """
        self.tracker = tracker
        self.parameters = parameters
        self.inputs = self._inputs_prepare()
        self.acceptance_ratio = None
    
    def _inputs_prepare(self) -> dict:
        """
        This function calculates other necessary inputs to the MILP optimizer, stored in a dictionary.

        Returns:
        inputs: dict
            Include self.parameters and other necessary inputs calculated out in this function.
        """
        def _f_generation(detour:np.array, price:np.array, urgency:float, size:int) -> tuple:
            """
            This function is used to generate the look up table for the f in the objective formula for linearization.

            Parameters:
            detour: Numpy array
                detour[i, j] indicates the detour distance between courier i and parcel j.
            price: Numpy array
                All the price[i, j] are the same, indicating all the possible price levels that can be proposed.
                For each price[i, j], there are k price levels, defined by configuration.parameters['price_levels_num'].
                Therefore, it is a 3-dimensional matrix.
            urgency: float
                The urgency of a parcel.
            size: int
                The size of the parcel.

            Returns:
            f: Numpy array
                The look up table of f.
                f[i, j, k] is the objective value calculated from the objective function, given detour[i, j],
                price[i, j, k] and the other parameters.
            acceptance: Numpy array
                It stores the acceptance probablity for all the f[i, j, k] entries.
            """
            num_couriers = len(self.tracker.idle_couriers)
            num_parcels = len(self.tracker.active_parcels)

            f = np.zeros((num_couriers, num_parcels, price.shape[2]))
            acceptance = np.zeros((num_couriers, num_parcels, price.shape[2]))

            for i in range(f.shape[0]):
                for j in range(f.shape[1]):
                    for k in range(f.shape[2]):
                        acceptance[i, j, k] = acceptance_ratio_calculation(price[i, j, k], detour[i, j], size[j])
                        f[i, j, k] = weight_calculation(price[i, j, k], detour[i, j], urgency[j], size[j], decay=1)
            return f, acceptance
        
        num_price_levels = self.parameters['price_levels_num']
        U = self.parameters['U'] # The upper bound of compenation rate.
        
        # compensation rate per Km
        num_couriers = len(self.tracker.idle_couriers)
        num_parcels = len(self.tracker.active_parcels)
        price_vector = np.linspace(0, U, num_price_levels) # k price levels for a price[i, j] entry.
        price_matrix = np.tile(price_vector, (num_couriers, num_parcels, 1)) # Duplicate the price vector to all the price[i, j] entries.
        idle_couriers_list = list(self.tracker.idle_couriers.values())
        active_parcels_list = list(self.tracker.active_parcels.values())
        urgency = np.zeros((num_parcels))
        size = np.zeros((num_parcels))
        detour = np.zeros((num_couriers, num_parcels))
        vc = np.zeros((num_couriers, num_parcels)) # vc=1 if a courier has enough capacity to carry a parcel.

        # Calculate the detour matrix and capacity constraints.
        for i in range(num_couriers):
            courier = idle_couriers_list[i]
            for j in range(num_parcels):
                parcel = active_parcels_list[j]
                detour[i, j] = get_detour(self.tracker, detour[i, j], courier, parcel)
                vc[i, j] = capacity_judgement(courier, parcel)
                if urgency[j] == 0: # urgency and parcel only need to be written once. This 'if' helps avoid 'i' loops.
                    urgency[j] = parcel.urgency
                    size[j] = parcel.size

        f, acceptance = _f_generation(detour, price_matrix, urgency, size)

        inputs = configuration.parameters
        inputs['num_couriers'] = num_couriers
        inputs['num_parcels'] = num_parcels
        inputs['price'] = price_matrix
        inputs['f'] = f
        inputs['urgency'] = urgency
        inputs['vc'] = vc
        inputs['acceptance'] = acceptance
        inputs['detour'] = detour

        return inputs

    def optimize(self) -> dict:
        """
        Solve the matching problem using MILP algorithm.

        Returns:
        A dictionary containing the result.
        """
        gurobi_key = keywords.gurobi_key
        
        def convert_from_gurobi(price:np.array, w: gp.Var, row_num:int, col_num:int, depth_num:int, acceptance:np.array) -> tuple:
            """
            This function converts selected prices, boolean price level selectiom variable (w) and assignments (x) 
            from Gurobi format to Numpy arrays.

            Parameters:
            price: Numpy array
                3D price matrix containing all the price levels.
            w: gb.Var
                Optimized result indicating which price level will be chosen.
            col_num, row_num, depth_num: int
                Dimensions of price and w.
                depth_num is only useful for price matrix, since only it is a 3D matrix.
            acceptance: Numpy array
                Precalculated acceptance array, used for filtering invalid assignments.
                Assignments with acceptance probability == 0 are ignored.

            Returns:
            result_w, result_x, result_p: Numpy array
                Noticeable, price matrix is changed from 3D to 2D.
                In price[i, j] there are k price levels, and one of them is selected by w[i, j, k].
                (Only one element in w[i, j] among k elements will be 1, meaning the corresponding
                price[i, j, k] is selected.) Therefore, the resulted price matrix (result_p) now
                is 2D, and only the selected price level is kept in result_p[i, j].
            """
            result_w = np.zeros((row_num, col_num, depth_num))
            result_x = np.zeros((row_num, col_num))
            result_p = np.zeros((row_num, col_num))

            for row in range(row_num):
                for col in range(col_num):
                    for depth in range(depth_num):
                        result_w[row, col, depth] = w[row, col, depth].X
                        # Only if the selected price != 0 will the assignment be made.
                        # w >= 0.9 is to avoid ignorances of w whose values are like 9.9e-1.
                        if w[row, col, depth].X >= 0.9 and acceptance[row, col, depth] != 0 : 
                            result_x[row, col] = 1
                            result_p[row, col] = price[row, col, depth]

            return result_w, result_x, result_p

        def accpetance_ratio_3D_to_2D(assignments:np.array, w:np.array, acceptance_3D:np.array) -> np.array:
            """
            Convert 3D acceptance matrix to 2D based on the result selections.
            acceptance_3D is a matrix containing the acceptance probability under all the assignments
            and offered price situations. The dimension of acceptance_3D is [i, j, k], and k elements
            correspond the k price levels. After assignments have been made, only one of them is 
            selected. Therefore the resulted acceptance matrix (acceptance_2D) only has 2 dimensions.

            Parameters:
            assignments, w: Numpy arrays
                The optimized results from MILP.
            acceptance_3D: Numpy array
                The 3D acceptance map from the inputs.

            Returns:
            acceptance_2D: Numpy array
                The converted acceptance dimension.
            """
            acceptance_2D = np.zeros((assignments.shape[0], assignments.shape[1]))
            for i in range(assignments.shape[0]):
                for j in range(assignments.shape[1]):
                    if assignments[i, j] >= 0.9:
                        for k in range(w.shape[2]):
                            if w[i, j, k] >= 0.9:
                                acceptance_2D[i, j] = acceptance_3D[i, j, k]
                                break
            return acceptance_2D

        num_couriers = self.inputs['num_couriers']
        num_parcels = self.inputs['num_parcels']
        num_levels = self.inputs['price_levels_num']
        price = self.inputs['price']
        f = self.inputs['f']
        vc = self.inputs['vc']
        U = self.inputs['U']
        acceptance = self.inputs['acceptance']

        env = gp.Env(params=gurobi_key)
        model = gp.Model(env=env)
        model.Params.OutputFlag = 0
        model.setParam('Threads', 1)

        x = model.addVars(num_couriers, num_parcels, vtype=GRB.BINARY) # Assignments.
        w = model.addVars(num_couriers, num_parcels, price.shape[2], lb=0)  # Price levels selection.
        v = model.addVars(num_couriers, num_parcels, price.shape[2] - 1, vtype=GRB.BINARY)  # Intermediate decision variable for linearization.

        for courier in range(num_couriers):
            for parcel in range(num_parcels):
                x[courier, parcel].start = 0 # Initialise x with 0.

        model.addConstrs(gp.quicksum(x[i, j] for i in range(num_couriers)) <= 1 for j in range(num_parcels))
        model.addConstrs(gp.quicksum(x[i, j] for j in range(num_parcels)) <= 1 for i in range(num_couriers))
        model.addConstrs(x[i, j] <= vc[i, j] for i in range(num_couriers) for j in range(num_parcels))
        model.addConstrs(gp.quicksum(w[i, j, k] * price[i, j, k] for k in range(price.shape[2])) <= U * x[i, j] for i in range(num_couriers) for j in range(num_parcels))
        model.addConstrs(w[i, j, 0] <= v[i, j, 0] for i in range(num_couriers) for j in range(num_parcels))
        model.addConstrs(w[i, j, k] <= v[i, j, k - 1] + v[i, j, k] for i in range(num_couriers) for j in range(num_parcels) for k in range(1, price.shape[2] - 1))
        model.addConstrs(w[i, j, price.shape[2] - 1] <= v[i, j, price.shape[2] - 2] for i in range(num_couriers) for j in range(num_parcels))
        model.addConstrs(gp.quicksum(v[i, j, k] for k in range(price.shape[2] - 1)) == 1 for i in range(num_couriers) for j in range(num_parcels))
        model.addConstrs(gp.quicksum(w[i, j, k] for k in range(price.shape[2])) == 1 for i in range(num_couriers) for j in range(num_parcels))

        model.setObjective(gp.quicksum(f[i, j, k] * w[i, j, k] for i in range(num_couriers) for j in range(num_parcels) for k in range(price.shape[2])), GRB.MINIMIZE)
        model.optimize()

        w, x, p = convert_from_gurobi(price, w, num_couriers, num_parcels, num_levels, acceptance)
        self.acceptance_ratio = accpetance_ratio_3D_to_2D(x, w, self.inputs['acceptance'])

        return {
            'assignments': x,
            'w': w,
            'prices': p,
            'objective_value': model.ObjVal
        }
    
class Hungarian:
    def __init__(self, parameters:dict, tracker:Tracker):
        """
        Parameters:
        parameters: dict
            A dictionary of hyperparameters for optimization.
        tracker: Tracker instance.
        """
        self.tracker = tracker
        self.parameters = parameters
        self.acceptance_ratio = None
    
    def _inputs_prepare(self) -> tuple:
        """
        This function is used to generate the cost matrix, which contains the best objective
        value for all the assignment pairs and the corresponding acceptance ratio and price.
        They are the essential inputs to optimize the problem.

        Returns:
        cost_matrix: Numpy array
            A matrix containing the best objective values for all the assignment pairs.
        selected_prices: Nump array
            A matrix containing the optimal prices.
        acceptance_ratio: Numpy array
            Corresponding acceptance probability.
        """
        # Get dimension information for the matrices.
        ax1_num = len(self.tracker.idle_couriers)
        ax2_num = len(self.tracker.active_parcels)

        selected_prices = np.zeros((ax1_num, ax2_num), dtype=float)
        detour = np.zeros((ax1_num, ax2_num), dtype=float)
        acceptance_ratio = np.zeros((ax1_num, ax2_num), dtype=float)
        cost_matrix = np.full((ax1_num+1, ax2_num+1), LARGE, dtype=float)

        # Add dummy rows and columns so that the optimizer can decide not to make any assignments.
        cost_matrix[-1,:] = 0
        cost_matrix[:, -1] = 0

        # Start calculating the best prices the corresponding objective values for all the assignment pairs.
        print('Building the graph...')
        idle_couriers_list = list(self.tracker.idle_couriers.values())
        active_parcels_list = list(self.tracker.active_parcels.values())

        with tqdm(total = len(idle_couriers_list) * len(active_parcels_list)) as pbar:
            for i in range(len(idle_couriers_list)):
                courier = idle_couriers_list[i]
                for j in range(len(active_parcels_list)):
                    parcel = active_parcels_list[j]
                    
                    if (courier.index, parcel.index) not in self.tracker.not_match: # Filter out the (courier, parcel) pairs that are marked as cannot be matched previously.
                        detour[i, j] = get_detour(self.tracker, detour[i, j], courier, parcel) # Calculate the detour distance.

                        # There are new edges connecting parcels with new couriers, which may not satisfy constraints but haven't been filtered out yet.
                        # Therefore the if the courier's capacity is not large enough or the detour is too large, this courier is filtered out.
                        if courier.capacity >= parcel.size and detour[i, j] <= configuration.assignments_detour_threshold:
                            optimal_price, optimal_weight = find_best_weight(parcel, detour[i, j])
                            acceptance_ratio[i, j] = acceptance_ratio_calculation(optimal_price, detour[i, j], parcel.size)
                            cost_matrix[i, j] = optimal_weight
                            selected_prices[i, j] = optimal_price
                        else: # If the new courier cannot be assigned to the current parcel due to the violation of constraints.
                            self.tracker.not_match[(courier.index, parcel.index)] = 1
                    pbar.update(1)

        return cost_matrix, selected_prices, acceptance_ratio

    def optimize(self):
        """
        This function optimizes the problem using the heuristic Hungarian method, 
        and returns a dict containing useful results.

        Returns:
        A dictionary containing information of assignments and the selected compensation rate.
        """
        
        cost_matrix, selected_prices, self.acceptance_ratio = self._inputs_prepare()

        ax1_num = len(self.tracker.idle_couriers)
        ax2_num = len(self.tracker.active_parcels)
        assignments = np.zeros((ax1_num, ax2_num))
        minimised_total_weight = 0.0

        print('Assigning the parcels...')
        assigned_couriers_indices, assigned_parcels_indices = linear_sum_assignment(cost_matrix) # Solve the problem using Hungarian algorithm.
        for i, j in zip(assigned_couriers_indices, assigned_parcels_indices):
            if cost_matrix[i, j] != LARGE and cost_matrix[i, j] != 0 and selected_prices[i, j] >= 0.1:
                assignments[i, j] = 1
                minimised_total_weight += cost_matrix[i, j]
                parcel = list(self.tracker.active_parcels.values())[j]
                parcel.best_weight = cost_matrix[i, j] # Record the best weight in parcel's attributes for further comparison.

        return {
            'assignments': assignments,
            'prices': selected_prices,
            'objective_value': minimised_total_weight
        }    

class Execution:
    def __init__(self, seed:int, tracker:Tracker, optimizer):
        """
        seed:int
            Random seed
        tracker: Tracker instance
        optimizer:
            MILP or Hungarian optimizer.
        """
        self.seed = seed
        self.tracker = tracker
        self.optimizer = optimizer
        self.result = None
        self.acceptance_ratio = None
        
    def solve(self) -> float:
        """
        Solves the matching problem.

        Returns:
        objective_value: float
            The derived objective value of the problem.
        """
        self.result = self.optimizer.optimize()
        self.acceptance_ratio = self.optimizer.acceptance_ratio
        return self.result['objective_value']

    def couriers_behave(self) -> list:
        """
        Given the acceptance probability of a courier accepting an assigned parcel, 
        this function generates an acceptance or rejection behavior.

        Returns:
        assigned_parcels: list
            A list of all the successfully assigned parcels.
        """
        assigned_parcels = []
        assigned_couriers = []

        idle_couriers_list = list(self.tracker.idle_couriers.values())
        active_parcels_list = list(self.tracker.active_parcels.values())

        for i in range(self.result['assignments'].shape[0]):
            t = tools.time_illustration(self.tracker.t)
            report_flag = 1

            for j in range(self.result['assignments'].shape[1]):
                if self.result['assignments'][i, j] == 1:
                    report_flag = 0
                    courier = idle_couriers_list[i]
                    parcel = active_parcels_list[j]

                    behaviour = acception.naive_accept(self.seed, self.acceptance_ratio[i, j])

                    # If the courier accepts the assignment.
                    if behaviour == True:
                        print('Parcel ' + str(parcel.index) + ' is assigned to the courier ' + str(courier.index) + ' at time ' + t.strftime('%H:%M') + ' .\n')

                        update.CourierUpdate.courier_accept(parcel, courier, self.result['prices'][i, j], self.acceptance_ratio[i, j])
                        update.ParcelsUpdate.parcel_accepted(self.tracker, parcel, courier, self.result['prices'][i, j])
                        
                        assigned_parcels.append(parcel)
                        assigned_couriers.append(courier)
                        # self.tracker._tot_compensation += self.result['prices'][i, j]

                    # If not.
                    else:
                        print('The courier ' + str(courier.index) + ' has rejected the assignment for the parcel ' + str(parcel.index) + ' at time ' + t.strftime('%H:%M') + '.\n')
                        update.CourierUpdate.courier_reject(parcel, courier, self.result['prices'][i, j], self.acceptance_ratio[i, j])
                    break
            
            # If there are no assignments for the courier.
            if report_flag:
                print(f"There are no assignments for the courier {idle_couriers_list[i].index} at time {t.strftime('%H:%M')} \n")

        # Maintenence of the lists.
        for parcel in assigned_parcels:
            self.tracker.active_parcels.pop(parcel.index)
            parcel.status = configuration.assigned
        for courier in assigned_couriers:
            self.tracker.idle_couriers.pop(courier.index)

        return assigned_parcels




