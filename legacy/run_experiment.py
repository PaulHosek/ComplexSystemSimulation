"""
This file can be used to run our CA model (outdated version).
"""

import numpy as np
import os
from initial_distributions import Create_Initial_Topography,order_distribution
from CA_model import CA_model

#Parameters:
mode = "order"                      #"order", "diffusion","reyleigh", "snow_dune"
size = 500                          #size of the grid in pixels 
control_parameter =  0.4            #level of uniformity of topology 0-1
iterations = 100_000              #number of states to evolve in time
experiment_name = "Test2"            #needed for storing states in folder
periodic = False                    #use periodic boundaries
dt = 0.1                            #time discretization
dx = 40                             #space discretization

#initialize topography
if mode == "order":
   topology = order_distribution(control_parameter, size)

else:
   topology = Create_Initial_Topography(res=size, mode=mode, tmax=2, dt=0.1, g=1, sigma_h=1., h=0., snow_dune_radius=1.,
                              Gaussians_per_pixel=0.2,
                              number_of_r_bins=150, window_size=5, snow_dune_height_exponent=1.)

#initialize empty water topology
h = np.zeros((size,size))

#initialize model with topology
model = CA_model(topology, h, dt, dx, periodic_bounds = periodic)

#create experiment folder
os.mkdir(f"experiments/{experiment_name}")

#create state folders
os.mkdir(f"experiments/{experiment_name}/pond")
os.mkdir(f"experiments/{experiment_name}/ice")

#run the experiment for number of iterations
for i in range(iterations):
   
   model.step()

   #save every 1000th state
   if i % 1000 == 0:
      
      np.save(f"experiments/{experiment_name}/pond/_i={i}",model.h)
      np.save(f"experiments/{experiment_name}/ice/_i={i}",model.Ht)

