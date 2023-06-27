import numpy as np
import os
from initial_distributions import Create_Initial_Topography,order_distribution
from CA_model import CA_model

#Parameters:
# mode = "snow_dune"                      #"order", "diffusion","reyleigh", "snow_dune"
# size = 500                          #size of the grid in pixels 
# control_parameter =  0.4            #level of uniformity of topology 0-1
# iterations = 100_000              #number of states to evolve in time
# experiment_name = "Test4plotting"            #needed for storing states in folder
# periodic = False                    #use periodic boundaries
# dt = 15                         #time discretization
# dx = 1                             #space discretization

# #initialize topography
# if mode == "order":
#    topology = order_distribution(control_parameter, size)

# else:
#    topology = Create_Initial_Topography(res=size, mode=mode, tmax=2, dt=0.1, g=1, sigma_h=1., h=0., snow_dune_radius=1.,
#                               Gaussians_per_pixel=0.2,
#                               number_of_r_bins=150, window_size=5, snow_dune_height_exponent=1.)

experiment_name = "Test4plotting"

# initialize model with 'snow dune topography' Popovic et al., 2020

res = 500                       # size of the domain
mode = 'snow_dune'              # topography type
tmax = 2; dt = 0.1              # diffusion time and time-step if mode = 'diffusion' or mode = 'rayleigh'
g = 1                           # anisotropy parameter
sigma_h = 0.03                  # surface standard deviation
snow_dune_radius = 1.           # mean snow dune radius if mode = 'snow_dune'  
Gaussians_per_pixel = 0.2       # density of snow dunes if mode = 'snow_dune'  
snow_dune_height_exponent = 1.  # exponent that relates snow dune radius and snow dune height if mode = 'snow_dune'

iterations = 1000

mean_freeboard = 0.1

Tdrain = 10.; dt_drain = 0.5    # time and time-step of to drainage

# create topography
Ht_0 = Create_Initial_Topography(res = res, mode = mode, tmax = tmax, dt = dt, g = g, sigma_h = sigma_h, h = mean_freeboard, snow_dune_radius = snow_dune_radius, 
            Gaussians_per_pixel = Gaussians_per_pixel, number_of_r_bins = 150, window_size = 5, snow_dune_height_exponent = snow_dune_height_exponent)


size = res
h = np.zeros(shape = (size, size))

#initialize empty water topology
h = np.zeros((size,size))

#initialize model with topology
model = CA_model(Ht_0, h, dt = 15, dx = 1, periodic_bounds = False)

#create experiment folder
if not os.path.exists(f"experiments/{experiment_name}"):
   os.mkdir(f"experiments/{experiment_name}")

#create state folders
if not os.path.exists(f"experiments/{experiment_name}/pond"):
   os.mkdir(f"experiments/{experiment_name}/pond")
if not os.path.exists(f"experiments/{experiment_name}/ice"):
   os.mkdir(f"experiments/{experiment_name}/ice")

#run the experiment for number of iterations
for i in range(iterations):
   
   model.step()

   #save every 1000th state
   if i % 1000 == 0:
      
      np.save(f"experiments/{experiment_name}/pond/_i={i}",model.h)
      np.save(f"experiments/{experiment_name}/ice/_i={i}",model.H)

