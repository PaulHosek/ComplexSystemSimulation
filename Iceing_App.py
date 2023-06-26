import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from iceing import iceing_model
from iceing import glauber
from iceing import perim_area
from matplotlib import colors

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
    This demo shows the formation of meltponds on arctic sea ice during the summer period using an adaptation from the Ising model.
    Start by selecting an initial fraction of meltponds and a topology and run the simulation.

    """
)

#create sliders and buttons
with st.sidebar:
    with st.form(key="simulation_param"):
        input_fraction = st.slider('Initial fraction of meltponds', 0.0, 1.0, 0.1)
        topology = st.selectbox('Initial topology', ["normal","snow dune","diffusion","kayleigh"])
        st.session_state.initialize = st.form_submit_button(label="Initialize domain")
        st.session_state.start_sim_clicked = st.form_submit_button(label="Start Simulation")
        st.session_state.stop_sim_clicked = st.form_submit_button(label ="Stop Simulation")

#Initialize domain
if st.session_state.initialize:
    with st.spinner('Initialize domain..'):
        #initialize model
        st.session_state.model = iceing_model(input_fraction,120)

        #display initial topology
        # st.session_state.fig,st.session_state.ax = plt.subplots(figsize=(5,5))

        st.session_state.fig = plt.figure(constrained_layout=True)
        st.session_state.axs = st.session_state.fig.subplot_mosaic([['Left', 'TopRight'],['Left', 'BottomRight']],
                          gridspec_kw={'width_ratios':[2, 1]})
        
        st.session_state.axs['Left'].set_title('Meltponds')
        st.session_state.axs['TopRight'].set_title('Sum of Ponds')
        st.session_state.axs['BottomRight'].set_title('Perimeters')

        st.session_state.axs['Left'].imshow(st.session_state.model.s,cmap=cmap,norm=norm)

        # #calculate the perim area and plot
        areas, perimeters = perim_area(st.session_state.model.s, pond_val = -1, ice_val = 1)
        st.session_state.axs['BottomRight'].scatter(areas,perimeters)

        st.pyplot(st.session_state.fig)

        
#Run the simulation once start button is clicked
if st.session_state.start_sim_clicked:
    with st.spinner("Running Simulation"):
            
            #stop simulation once stop button is clicked
            with st.empty():
                while not st.session_state.stop_sim_clicked:

                    #run the simulation for x timesteps
                    new_state,st.session_state.sums = glauber(st.session_state.model.s,st.session_state.model.hi,800)
                    st.session_state.model.s = new_state

                    #display new state
                    # st.session_state.fig,st.session_state.ax = plt.subplots(figsize=(5,5))
                    # plt.imshow(st.session_state.model.s,cmap=cmap,norm=norm)
                    # st.pyplot(st.session_state.fig,clear_figure=True)

                    # #calculate the perim area and plot
                    # areas, perimeters = perim_area(new_state, pond_val = -1, ice_val = 1)
                    # print(areas,perimeters)
                    # st.pyplot(plt.scatter(areas, perimeters))

                    # st.session_state.fig = plt.figure(constrained_layout=True)
                    # axs = st.session_state.fig.subplot_mosaic([['Left', 'TopRight'],['Left', 'BottomRight']],
                    #                 gridspec_kw={'width_ratios':[2, 1]})
                    
                    # axs['Left'].set_title('Meltponds')
                    # axs['TopRight'].set_title('Sum of Ponds')
                    # axs['BottomRight'].set_title('Perimeters')

                    st.session_state.axs['Left'].imshow(st.session_state.model.s,cmap=cmap,norm=norm)

                    #calculate the perim area and plot
                    areas, perimeters = perim_area(st.session_state.model.s, pond_val = -1, ice_val = 1)
                    st.session_state.axs['BottomRight'].scatter(areas,perimeters)
                    st.session_state.axs["TopRight"].plot(st.session_state.sums)

                    #plot the sum of ponds
                    st.pyplot(st.session_state.fig)


    #plot latest state
    st.session_state.fig = plt.figure(constrained_layout=True)
    st.session_state.axs = st.session_state.fig.subplot_mosaic([['Left', 'TopRight'],['Left', 'BottomRight']],
                          gridspec_kw={'width_ratios':[2, 1]})
    st.session_state.axs['Left'].imshow(st.session_state.model.s,cmap=cmap,norm=norm)

    st.session_state.axs['Left'].set_title('Meltponds')
    st.session_state.axs['TopRight'].set_title('Sum of Ponds')
    st.session_state.axs['BottomRight'].set_title('Perimeters')

    #calculate the perim area and plot
    areas, perimeters = perim_area(st.session_state.model.s, pond_val = -1, ice_val = 1)
    st.session_state.axs['BottomRight'].scatter(areas,perimeters)
    st.session_state.axs["TopRight"].plot(st.session_state.sums)

    #plot the sum of ponds
    st.pyplot(st.session_state.fig)

