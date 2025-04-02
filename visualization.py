import matplotlib.pyplot as plt
from datetime import time
from copy import deepcopy
import numpy as np
import configuration
import glob
import os
import seaborn as sns
import pandas as pd

def random_color(i:int) -> tuple:
    """
    Given a value, this function returns a corresponding color in RGB format.

    Parameters:
    i: int
        To which a color will be assigned.

    Returns:
    Tuple(float, float, float):
        A color in RGB format.
    """
    i += 1
    return (i * 0.878 % 1, i * 0.698 % 1, i * 0.259 % 1)

def plot_assignments(couriers_dic:dict, parcels_dic:dict, t:time, loc:str):
    """
    Plot the assignments among couriers and parcesl.
    Then save the plotted images.

    Parameters:
    couriers_dic, parcels_dic: dictionary
        Two dictionaries containing couriers and parcels to plot.
    t: time
        The corresponding time.
    loc: str
        The folder to save the figure.
    """
    couriers = couriers_dic.values()
    parcels = parcels_dic.values()

    x_parcel = [parcel.location[0] for parcel in parcels]
    y_parcel = [parcel.location[1] for parcel in parcels]
    color_parcel = [parcel.status for parcel in parcels]

    x_courier = [courier.destination[0] for courier in couriers]
    y_courier = [courier.destination[1] for courier in couriers]
    color_courier = [courier.status for courier in couriers]

    fig, ax = plt.subplots(figsize=(30, 20))
    ax.scatter(x_parcel, y_parcel, color=color_parcel, s=70, label='Parcel Destinations')
    ax.scatter(x_courier, y_courier, color=color_courier, s=70, label='Courier Destinations', marker='x', linewidths=2)
    ax.scatter(0, 0, edgecolors=(0.322, 0.322, 0.322), facecolors='none', s=70, label='Locker')

    for parcel in parcels:
        if parcel.carried_by != '-' and parcel.carried_by != 'punished':
            arrow_color = random_color(parcel.carried_by)
            ax.annotate("",
                        xy=(parcel.location[0], parcel.location[1]), xycoords='data',
                        xytext=(0, 0), textcoords='data',
                        arrowprops=dict(arrowstyle="->", lw=1.2, color=arrow_color))
        
            courier = couriers_dic[parcel.carried_by]
            ax.annotate("",
                        xy=(courier.destination[0], courier.destination[1]), xycoords='data',
                        xytext=(parcel.location[0], parcel.location[1]), textcoords='data',
                        arrowprops=dict(arrowstyle="->", lw=1.2, color=arrow_color))
            ax.annotate("",
                        xy=(courier.destination[0], courier.destination[1]), xycoords='data',
                        xytext=(0, 0), textcoords='data',
                        arrowprops=dict(arrowstyle="-", linestyle="--", lw=1.2, color=arrow_color))

    ax.set_title(f"Time = {t.strftime('%H:%M')}", fontsize=14, fontweight='bold', loc='left')

    # Preparing parcel properties for annotations
    parcel_indices = [parcel.index for parcel in parcels]
    parcel_sizes = [parcel.size for parcel in parcels]
    parcel_carried_by = [parcel.carried_by for parcel in parcels]
    parcel_urgency = [parcel.urgency for parcel in parcels]
    parcel_time_window = [deepcopy(parcel.time_window) for parcel in parcels]
    parcel_detour = [parcel.accepted_detour for parcel in parcels]
    parcel_decay = [parcel.decay for parcel in parcels]
    for window in parcel_time_window:
        for i in range(len(window)):
            hours = window[i] // 60
            minutes = window[i] % 60
            window[i] = time(hour=hours, minute=minutes).strftime('%H:%M')
    parcel_properties = [[parcel_indices[i], parcel_sizes[i], parcel_carried_by[i], parcel_time_window[i], parcel_urgency[i], parcel_detour[i], parcel_decay[i]] for i in range(len(parcel_indices))]
    parcel_properties_names = ['Parcel', 'Size', 'Carried by', 'Time Window', 'Urgency', 'Accepted detour', 'Decay']

    # Annotations for Parcels
    for i in range(len(x_parcel)):
        text_lines = [f"{name}: {parcel_properties[i][j]}" for j, name in enumerate(parcel_properties_names)]
        text = "\n".join(text_lines)
        ax.annotate(text, (x_parcel[i], y_parcel[i]), textcoords="offset points", xytext=(0,10), ha='center')

    # Preparing courier properties for annotations
    courier_indices = [courier.index for courier in couriers]
    capacity = [courier.capacity for courier in couriers]
    carrying = [courier.carrying for courier in couriers]
    rejected_offers = [courier.rejected_offers for courier in couriers]
    accepted_offers = [courier.accepted_offers for courier in couriers]
    courier_properties = [[courier_indices[i], capacity[i], carrying[i], accepted_offers[i], rejected_offers[i]] for i in range(len(courier_indices))]
    courier_properties_names = ['Courier', 'Capacity', 'Carrying', 'Accepted_Offers', 'Rejected Offers']

    # Annotations for couriers
    for i in range(len(x_courier)):
        text_lines = [f"{name}: {courier_properties[i][j]}" for j, name in enumerate(courier_properties_names)]
        text = "\n".join(text_lines)
        ax.annotate(text, (x_courier[i], y_courier[i]), textcoords="offset points", xytext=(0,10), ha='center')

    ax.legend()

    plt.savefig(loc)
    plt.close(fig)

def plot_ratio_couriers(tracker, save_loc:str, start_time:float, end_time:float):
    """
    Plot some interesting couriers features in one figure for comparison.

    Parameters:
    tracker: Tracker instance
    save_loc: str
        Location for saving the figure.
    start_time, end_time: float
        The start time and the end time for the plot.
    """
    idle_couriers_num = tracker._idle_couriers_num
    new_couriers_num = tracker._new_couriers_num
    assigned_couriers_num = tracker._assigned_couriers_num
    logout_couriers_num = tracker._logout_couriers_num

    _, ax = plt.subplots(figsize=(15, 8))
    x = np.arange(start_time, end_time, 1)

    ax.plot(x, idle_couriers_num, label='Idle couriers amount')
    ax.plot(x, new_couriers_num, label='New couriers amount')
    ax.plot(x, assigned_couriers_num, label='Assigned couriers amount')
    ax.plot(x, logout_couriers_num, label='Logged out couriers amount')

    ax.set_xlabel('Time')
    ax.set_ylabel('Counts Amount')
    ax.set_title('Various Couriers Amounts at the End of the Time Step')
    ax.legend()

    plt.savefig(save_loc, dpi=300)

def plot_ratio_parcels(tracker, save_loc:str, start_time:float, end_time:float):
    """
    Plots various parcels features for comparisons.

    Parameters:
    tracker: Tracker instance
    save_loc: str
        The location for saving the figure.
    start_time, end_time: float
        The simulation start and end time.
    """
    active_parcels_num = tracker._active_parcels_num
    assigned_parcels_num = tracker._assigned_parcels_num
    expired_parcels_num = tracker._expired_parcels_num
    standby_parcels_num = [configuration.parcels_num - a - b for a, b in zip(assigned_parcels_num, expired_parcels_num)]

    _, ax = plt.subplots(figsize=(15, 8))
    x = np.arange(start_time, end_time, 1)

    ax.plot(x, active_parcels_num, label='Active parcels amount')
    ax.plot(x, assigned_parcels_num, label='Assigned parcels amount')
    ax.plot(x, expired_parcels_num, label='Expired parcels amount')
    ax.plot(x, standby_parcels_num, label='Total parcels left for assignments')

    ax.set_xlabel('Time')
    ax.set_ylabel('Parcels Amount')
    ax.set_title('Comparison of Various Parcels Features')
    ax.legend()
    plt.savefig(save_loc, dpi=300)

def plot_lines_compare(x:np.array, lines:list, labels:list, loc:str):
    """
    Plot each row in an np array as a line for comparison.

    Parameters:
    x: np.array
        The array for x axis.
    lines: list
        A list containing the data to be plotted.
    labels: list
        A list containing names for each row in the 'lines'.
    loc: str
        The location to save the figure.
    """
    _, ax = plt.subplots(figsize=(10, 5))
    for i in range(len(lines)):
        ax.plot(x, lines[i], label=labels[i])
    plt.legend()
    plt.savefig(loc)

def time_window_visualise(tracker, loc:str):
    """
    Visualizes the distribution of time windows of all the parcels.

    Parameters:
    tracker: Tracker instance
    loc: str
        The location of the file.
    """
    availability = np.zeros((configuration.end_time-configuration.start_time))
    for parcel in tracker.parcels.values():
        start, end = parcel.time_window
        availability[int(start-configuration.start_time) : int(end-configuration.start_time)] += 1
        
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(configuration.start_time, configuration.end_time, 1), availability, drawstyle='steps-post')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Number of Available Parcels')
    x_ticks = np.arange(configuration.start_time, configuration.end_time + 1, 60)
    x_labels = [f"{int(t // 60):02d}:{int(t % 60):02d}" for t in x_ticks]
    plt.xticks(x_ticks, x_labels, rotation=45)
    plt.title('Parcel Availability Over Time')
    plt.grid(True)
    plt.savefig(loc)

def clear_folder(folder:str):
    """ 
    Deletes all files in the specified folder.

    Parameters:
    folder: str
        The name of the folder.
    """
    files = glob.glob(f'{folder}/*')
    for f in files:
        os.remove(f)

def visualize_assignments(filename:str):
    """
    Visualize the simulation log to exhibit assignments and other detailed information.

    Parameters:
    filename: str
        The name of the file in which the simulation log is storaged.
    """
    os.makedirs(os.path.dirname('main/output/images/assignments/'), exist_ok=True)
    clear_folder('main/output/images/assignments/')
    log = np.load(f'main/output/logs/{filename}', allow_pickle=True)

    for _, (tracker, time_) in enumerate(log): # log format: [(tracker1, t1), (tracker2, t2), ...]
        hours = time_ // 60
        minutes = time_ % 60
        t = time(hour=hours, minute=minutes)
        plot_assignments(tracker.couriers, tracker.parcels, t, f'main/output/images/assignments/{t}.png')

def parcels_costs_analyse(parcels:list, loc:str):
    """
    Plots three figures illustrating contributions to parcels' costs. The first figure is a scatter, 
    illustrating the relationship among costs, detour, size and urgency. The second figure reveals
    the correlation of the cost with parcels' size, urgency and detour. The last figure shows
    The frequency of the accepted orders' compensation in each range.

    Parameters:
    parcels: dict
        A list of parcels to be analyzed here.
    loc: str
        The location of the folder for saving the figures.
    """
    def urgency_to_color(urgency:float) -> tuple:
        """
        Translates urgency to color. The more urgent a parcel is,
        the more red it is in the figure.
        """
        r = min(1.0, urgency)
        g = min(1.0, 1 - urgency)
        return (r, g, 0, 0.4)
    
    costs = [parcel.accepted_price for parcel in parcels]
    detours = [parcel.accepted_detour for parcel in parcels]
    weights = [parcel.size for parcel in parcels]
    urgencies = [parcel.urgency for parcel in parcels]
    colors = [urgency_to_color(urgency) for urgency in urgencies]

    plt.figure(figsize=(20, 12))
    plt.scatter(detours, costs, s=[weight * 50 for weight in weights], c=colors, alpha=0.4)
    plt.xlabel('Detour')
    plt.ylabel('Cost')
    plt.title('Parcel Cost VS Detour and Urgency')
    plt.grid(True)
    plt.savefig(f"{loc}cost_analysis_scatter.jpg", dpi=300)

    data = {
    'accepted_price': [parcel.accepted_price for parcel in parcels],
    'accepted_detour': [parcel.accepted_detour for parcel in parcels],
    'size': [parcel.size for parcel in parcels],
    'urgency': [parcel.urgency for parcel in parcels]
    }
    df = pd.DataFrame(data)
    correlations = df.corr()['accepted_price'].drop('accepted_price')
    plt.figure(figsize=(6, 4))
    sns.heatmap(correlations.to_frame(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation of accepted_price with other features')
    plt.savefig(f"{loc}cost_analysis_correlation.jpg", dpi=500)

    plt.figure(figsize=(16, 12))
    bins = np.arange(0, configuration.parameters['U'], 10)
    plt.hist(df['accepted_price'], bins=bins, alpha=0.7)
    plt.xlabel('Accepted Price')
    plt.ylabel('Frequency')
    plt.title('Histogram of Accepted Prices')
    plt.xticks(bins)
    plt.savefig(f"{loc}cost_analysis_histogram.jpg", dpi=500)