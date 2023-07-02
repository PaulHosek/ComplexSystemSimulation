# ComplexSystemSimulation

#### Introduction
This repository contains code for the UVA Master course *Complex Systems Simulation*. The aim of the project is to model melt ponds on the arctic sea ice. During summer the arctic ice starts to melt forming ponds of clear water depending on the topography of the ice. Meltponds increase the melt-rate, since water has a lower albedo ( 5-22% level of reflection) than ice (80-95%). This is called the 'albedo feedback mechanism'. Eventually the individual meltponds will merge forming complex and percolating clusters.

![alt text](https://github.com/PaulHosek/ComplexSystemSimulation/blob/main/Figures/The%20albedo%20effect_Perovich.jpg)

Two different model approaches were used. The first is an adaptation of the Ising model that simply redistributes an initial fraction of meltponds (Iceing). The second is a more physics informed model that models the vertical and horizontal flow of melt water (CA_model).

### Installation
The project has been published to PyPi and can be installed by:

`pip install ComplexSystemSimulation`

#### How to run

The simple Iceing model uses a web interface to run the simulations. The interface can be opened by running the following command:

`streamlit run web_interface.py`

The physics informed model can be run by opening the 'run_experiment.py' file and changing the parameters accordingly. Make sure to use a new experiment name. The model is run by:

`python3 run_experiment.py`

The states of the experiment will be saved to experiment_data/[experimentname] as .npy files. Which can be loaded and analysed later.


#### Structure

```
├── .idea                                           --> Contains the configuration of the project
│   ├── ComplexSystemSimulation.iml
├── build                                           --> Contains the build for pip installing
│   ├── bdist.linux-x86_64 
│   ├── lib
│   │   ├── CA_model.py                                 --> The functions for the physics informed meltpond model
│   │   ├── evaluation.py                               --> Functions for doing statistical analyis on the system
│   │   ├── Iceing_model.py                             --> Functions that define the Ising model
│   │   ├── topography.py                               --> Several initial ice sheet topographies
│   │   ├── web_interface.py                            --> web interface of the Ising model
├── dist                                            --> Contains the compressed project for PyPi
│   ├── ComplexSystemSimulation-1.0.0-py3-none-any.whl  --> Compressed project whl
│   ├── ComplexSystemSimulation-1.0.0.tar.gz            --> Compressed project tar.gz
├── experiment_data                                 --> Saved states of experiments as npy files
│   ├── h_normal_500px_250000iter.npy
│   ├── h_snow_dune_500px_25000iter_enhanced_melt_false.npy
│   ├── h_snow_dune_500px_25000iter_hor_flux_False.npy
│   ├── h_snow_dune_500px_25000iter_ice_melt_False.npy
│   ├── h_snow_dune_500px_25000iter_perbounds_False.npy
│   ├── h_snow_dune_500px_25000iter_seepage_False.npy
│   ├── h_snow_dune_500px_25000iter.npy
├── Figures                                         --> Figures of experiments and README images
│   ├── The albedo effect_Perovich.jpg
├── legacy                                          --> Old model versions and tests
│   ├── CA_luetje.ipynb     
│   ├── CA_model_outdated.py
│   ├── icesing.ipynb
│   ├── model_example.ipynb
│   ├── simple_CA.ipynb
├── papers                                          --> Research papers the model is based on
│   ├── 3D_arctic_melt_ponds.pdf
│   ├── GoldNAMSsupprefsNov2020.pdf
│   ├── Ma_2019_Ising_meltponds.pdf
│   ├── Ma_2019_supp_Ising_meltponds.pdf
│   ├── noaa_33437_DS1.pdf
│   ├── summer_evolution_melt_ponds.pdf
├── UnitTests                                       --> Tests for asserting correctness of code
│   ├── CAModelTest.py
│   ├── DetectPercolationTest.py
├── .gitignore                                      --> Intentionally untracked files on GIT
├── CA_model.py                                     --> Functionality of the physics informed model                                    
├── evaluation.py                                   --> Functions for analysis and information extraction
├── Iceing_model.py                                 --> Functionality of the simple Ising model
├── inflection_experiments                          --> Investigations of the fractal dimension phase transition
├── README.md                                       --> Overview of the project
├── requirements.txt                                --> List of requirements for pip install 
├── run_experiment.py                               --> Runs an experiment and saves each individual state of the system
├── setup.py                                        --> Setup file for pip install
├── topography.py                                   --> Initial ice sheet topographies
├── web_interface.py                                --> Web interface for the Iceing_model
```

## Models

### Iceing_model

This is an adaptation of the famous Ising model used to model the fractal dimension of arctic melt ponds.

The model is initialized randomly with a certain input fraction of meltwater 'F_in' and ice and an underlying ice sheet topography. At each timestep each cell observes their Von Neumann Neighborhood and changes its state (water or ice) according to the majority of the neighbor states, i.e. when three out of four neighbors are water, then the cell changes to water as well. If the neighbors are inconclusive,i.e. an equal number of water and ice neighbors, then the state of the cell is determined by a 'local magnetisation'. The local magnetisation is determined by the underlying ice sheet topography.

<p align="center">
<img src="https://github.com/PaulHosek/ComplexSystemSimulation/blob/main/Figures/Ising.jpeg"  width="300" height="300">
</p>



### CA_model

This is a physics informed Cellular Automata (CA). The model is adapted from Lüthje et al. 2006. The simulation is initialized with an ice sheet topography. At each time step a fraction of the ice is melted according to a fixed melt rate. Cells that already contain meltwater have increased melt rate due to the albedo feedback mechanism. Subsequently the meltwater is distributed across neighboring cells based on the gradient of the topography. Additionally some meltwater seeps vertically through the porous ice.

The change in meltwater is evolved according to:

$$\frac{\partial h}{\partial t}=He(h)\left(-s+\frac{\rho_{\text {ice }} \cdot m}{\rho_{\text {water }}}-\frac{g \rho_{\text {water }}}{\mu} \Pi_h \nabla \cdot(h \nabla \Psi)\right)$$

The equation for the evolution of sea-ice surface height $H_t$ , and hence topography, is given by

$$\frac{\partial H_t}{\partial t}=\frac{\partial \Psi}{\partial t}-\frac{\partial h}{\partial t}=He(H)(-m)$$

,where the melt-rate $m$ is given by 

$$
m=E m_i,
$$

,where

$$
\begin{gathered}
E=\left(1+\frac{m_p}{m_i} \frac{h}{h_{\max }}\right) 0 \leq h \leq h_{\max } \\
E=1+\frac{m_p}{m_i} h>h_{\max }
\end{gathered}
$$

Examples of the discrete domain can be found in the figures below


Discrete Schematic             |  3D Topography Example
:-------------------------:|:-------------------------:
![](https://github.com/PaulHosek/ComplexSystemSimulation/blob/main/Figures/3D_schematic.png)  |  ![](https://github.com/PaulHosek/ComplexSystemSimulation/blob/main/Figures/3D_topography.jpeg)


## Topography

The behaviour of the meltponds is very much dependent on the initial topography of the ice. Therefore experiments were performed with several different initial topographies. Below shows 

Snow Dune            |  Rayleigh |  Diffusion
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/PaulHosek/ComplexSystemSimulation/blob/main/Figures/Topography/3D/Topography_snow_dune_size_50.png)  |  ![](https://github.com/PaulHosek/ComplexSystemSimulation/blob/main/Figures/Topography/3D/Topography_rayleigh_size_50.png) |  ![](https://github.com/PaulHosek/ComplexSystemSimulation/blob/main/Figures/Topography/3D/Topography_diffusion_size_50.png)
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/PaulHosek/ComplexSystemSimulation/blob/main/Figures/Topography/2D/Topography_snow_dune_size_50.png)  |  ![](https://github.com/PaulHosek/ComplexSystemSimulation/blob/main/Figures/Topography/2D/Topography_rayleigh_size_50.png) |  ![](https://github.com/PaulHosek/ComplexSystemSimulation/blob/main/Figures/Topography/2D/Topography_diffusion_size_50.png)