"""
In this file, we develop a web interface to illustrate the capabilities of the icing-model and how a
change in the model-parameters or in the surface topography affects the overall end result.
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from Iceing_model import iceing_model
from Iceing_model import glauber
from Iceing_model import perim_area
from matplotlib import colors
from topography import Create_Initial_Topography

st.set_page_config(
    page_title="Iceing Model",
    page_icon="❄️",
    layout="wide",
)

cmap = colors.ListedColormap(['blue', 'white'])
bounds=[-100,0,100]
norm = colors.BoundaryNorm(bounds, cmap.N)

#set column containers for the graphs
col1, col2 = st.columns(2)

#set title and subtitle
st.title('Iceing Model')
st.write(
    """ 
    This demo shows the formation of meltponds on arctic sea ice during the summer period using an adaptation of the Ising model.
    Start by selecting an initial fraction of meltponds and a topography and run the simulation.
    
    The figure on the left shows the current state of the system where blue indicates water and white inidicates ice.
    The top figure to the right tracks the convergence of the system. It is the sum of the system states (-1,ice;+1,water). Once the sum flatlines, the system is in equilibrium and the simulation can be stopped.
    The bottom right figure shows the area and perimeter of each meltpond cluster in the system. It is an indication of the fractal dimension.

    """
)

def plot_state():
    """plots the state of the Iceing system"""
        
    #display initial topography
    st.session_state.fig = plt.figure(constrained_layout=True)
    st.session_state.axs = st.session_state.fig.subplot_mosaic([['Left', 'TopRight'],['Left', 'BottomRight']],
                        gridspec_kw={'width_ratios':[2, 1]})
    
    #titles
    st.session_state.axs['Left'].set_title('Melt Pond Topography')
    st.session_state.axs['TopRight'].set_title('Convergence')
    st.session_state.axs['BottomRight'].set_title('Perimeters')

    #formatting
    st.session_state.axs['TopRight'].set_title('Convergence')
    st.session_state.axs['TopRight'].set_xlabel('Iterations')
    st.session_state.axs['TopRight'].set_ylabel('Sum of ponds')
    st.session_state.axs['BottomRight'].set_yscale('log')
    st.session_state.axs['BottomRight'].set_xscale('log')
    st.session_state.axs['BottomRight'].set_xlabel("Area")
    st.session_state.axs['BottomRight'].set_ylabel("Perimeter")

    #remove labels and ticks from left plot
    st.session_state.axs["Left"].tick_params(left=False,right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    
    st.session_state.axs['Left'].imshow(st.session_state.model.s,cmap=cmap,norm=norm)

    #calculate the perim area and plot
    areas, perimeters = perim_area(st.session_state.model.s, pond_val = -1, ice_val = 1)
    st.session_state.axs['BottomRight'].scatter(areas,perimeters,marker=".")

    #plot convergence measure
    st.session_state.axs["TopRight"].plot(st.session_state.convergence)

    #plot final figure
    st.pyplot(st.session_state.fig)


#create sliders and buttons
with st.sidebar:
    with st.form(key="simulation_param"):
        input_fraction = st.slider('Initial fraction of meltponds', 0.0, 1.0, 0.45)
        topography = st.selectbox('Initial topography', ["normal","snow dune","diffusion","rayleigh"])
        st.session_state.initialize = st.form_submit_button(label="Initialize domain")
        st.session_state.start_sim_clicked = st.form_submit_button(label="Start Simulation")
        st.session_state.stop_sim_clicked = st.form_submit_button(label ="Stop Simulation")

#Initialize domain
if st.session_state.initialize:
    with st.spinner('Initialize domain...'):
        #initialize model
        st.session_state.model = iceing_model(input_fraction,120)

        #list for checking convergence values
        st.session_state.convergence = []
        st.session_state.convergence.append(np.sum(st.session_state.model.s))

        if topography == "snow dune":
            st.session_state.model.hi = Create_Initial_Topography(res=120, mode='snow_dune', tmax=2, dt=0.1, g=1, sigma_h=1., h=0., snow_dune_radius=1.,
                              Gaussians_per_pixel=0.2, number_of_r_bins=150, window_size=5, snow_dune_height_exponent=1.)
        elif topography == "diffusion":
            st.session_state.model.hi = Create_Initial_Topography(res=120, mode='diffusion', tmax=2, dt=0.1, g=1, sigma_h=1., h=0., snow_dune_radius=1.,
                              Gaussians_per_pixel=0.2, number_of_r_bins=150, window_size=5, snow_dune_height_exponent=1.)
        elif topography == "rayleigh":
            st.session_state.model.hi = Create_Initial_Topography(res=120, mode='rayleigh', tmax=2, dt=0.1, g=1, sigma_h=1., h=0., snow_dune_radius=1.,
                              Gaussians_per_pixel=0.2, number_of_r_bins=150, window_size=5, snow_dune_height_exponent=1.)

        #display initial state
        plot_state()

        
#Run the simulation once start button is clicked
if st.session_state.start_sim_clicked:
    with st.spinner("Running Simulation"):
            
            #stop simulation once stop button is clicked
            with st.empty():
                while not st.session_state.stop_sim_clicked:

                    #run the simulation for x timesteps
                    new_state,st.session_state.sums = glauber(st.session_state.model.s,st.session_state.model.hi,800)
                    st.session_state.convergence.append(np.sum(new_state))
                    st.session_state.model.s = new_state

                    #display new state
                    plot_state()


    #plot latest state
    plot_state()

