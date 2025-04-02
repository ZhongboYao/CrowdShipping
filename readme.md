# Description
The project simulates scenarios that when parcels are stored in lockers in public stations, the public
is offered some compensation for helping carry parcels to their destinations before going home. Receivers
have specified time windows in which the parcels are prefered to be delivered. Parcels that are failed
to be delivered within the time windows will result in penalty.

## Table of Contents
- [Folders](#folders)
- [Functional Files](#functional-files)
- [Executional Files](#executional-files)
- [Authors](#authors)

# Folders
## input
The folder contains data of arrival population at three stations, regression models for predicting the arrival population each customized time
step and figures for visualization.
## output
The 'Images' folder in the output folder contains images plotted by the simulation project. Another folder, named as 'Logs', contains simulations logs and assignment logs for understanding and analyzing what happend after simulation.
## setup
The folder contains couriers and parcles' destinations coordinates after the problem initialization.

# Funtional Files
## acception.py
This file contains functions of deciding whether the assignments will be accepted according to the given probability.
## configuration.py
The file contains all the hyper-parameters for the project.
## instances.py
In this file, Parcel, Courier class is defined. Tracker class is also defined, which is used to track the simulation status and is updated every time step.
## matching.py
Algorithms including MILP and Hungarian are included in this file. They are used for optimizing the problem. After optimization and assignments, whether couriers will accept the offers are also decided in this file.
## properties.py
This file conatins all the functions used to update or calculate values that are relavent to couriers' or parcels' attributes.
## simulation.py
A simulator is defined in this file, specifing how the simulation goes. It includes problem initialization, matching & optimization and saving & analyzing the results.
## tools.py
This file contains some other functions that are not that important. They are about saving the results, calculating some values like distances, etc.
## update.py
Functions in this file handles the update of parcels' and couriers' information. It includes their information changing with time, expiration, and changes after assignments.
## visulization.py
This file contains functions of visualizing the result of better analysis.

# Executional Files
## decayfuture.py
This file is the draft version of codes studying the efficienty of the algorithms considering the dynamics of the problem. 
## ratioplot.py
Functions in this file plot how the number and status of the parcels and couriers change during the simuation for better comparison and understanding. A simple simulation is also included in this file for just checking whether the simulation successfully works.

# Authors
- **Zhongbo Yao** - *Licentiate PhD Student at KTH*
- **Michele D Simonis** - *Assistant Professor at KTH* 