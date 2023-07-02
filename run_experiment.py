"""
This file can be used to run our CA model (outdated version).
"""

import numpy as np
import os
from topography import Create_Initial_Topography,order_distribution
from CA_model import CA_model

#Parameters:
mode = "snow_dune"                  #"order", "diffusion","reyleigh", "snow_dune"
size = 200                          #size of the grid in pixels 
control_parameter =  0.4            #level of uniformity of topology 0-1
iterations = 100_000                #number of states to evolve in time
experiment_name = "snow_dune_evolution"           #needed for storing states in folder
periodic = False                    #use periodic boundaries
tmax = 2; dt_top = 0.1              #diffusion time and time-step if mode = 'diffusion' or mode = 'rayleigh'
g = 1                               #anisotropy parameter
sigma_h = 0.03                      #surface standard deviation
snow_dune_radius = 1.               #mean snow dune radius if mode = 'snow_dune'  
Gaussians_per_pixel = 0.2           #density of snow dunes if mode = 'snow_dune'  
snow_dune_height_exponent = 1.      #exponent that relates snow dune radius and snow dune height if mode = 'snow_dune'
Tdrain = 10.; dt_drain = 0.5        #time and time-step of to drainage
dt = 15                             #model time discretization
dx = 1                              #model space discretization


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
os.mkdir(f"experiment_data/{experiment_name}")

#create state folders
os.mkdir(f"experiment_data/{experiment_name}/pond")
os.mkdir(f"experiment_data/{experiment_name}/ice")

#run the experiment for number of iterations
for i in range(iterations):
   
   model.step()

   #save every 1000th state
   if i % 1000 == 0:
      
      np.save(f"experiment_data/{experiment_name}/pond/_i={i}",model.h)
      np.save(f"experiment_data/{experiment_name}/ice/_i={i}",model.Ht)

