import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from iceing import iceing_model
from iceing import glauber


#set title and subtitle
st.header('Iceing Model')
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
        st.session_state.model = iceing_model(input_fraction,50)

        #display initial topology
        st.session_state.fig,st.session_state.ax = plt.subplots(figsize=(10,10))
        plt.imshow(st.session_state.model.s)
        st.pyplot(st.session_state.fig)

        
#Run the simulation once start button is clicked
if st.session_state.start_sim_clicked:
    with st.spinner("Running Simulation"):

        #stop simulation once stop button is clicked
        while not st.session_state.stop_sim_clicked:

            #run the simulation for x timesteps
            new_state,sums = glauber(st.session_state.model.s,st.session_state.model.hi,10000)

            st.session_state.model.s = new_state

            #display new state
            st.session_state.fig,st.session_state.ax = plt.subplots(figsize=(10,10))
            plt.imshow(st.session_state.model.s)
            st.pyplot(st.session_state.fig)

    #plot latest state
    st.session_state.fig,st.session_state.ax = plt.subplots(figsize=(10,10))
    plt.imshow(st.session_state.model.s)
    st.pyplot(st.session_state.fig)

