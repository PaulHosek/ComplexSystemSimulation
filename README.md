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



#### Structure

- `papers/...`: Inside this folder, you can find the main research papers that we used for inspiration when building this project.
- `UnitTests`: 
