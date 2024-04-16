import streamlit as st
import numpy as np
import pandas as pd
import pickle
import time
from matplotlib import pyplot as plt
from  matplotlib.ticker import FuncFormatter
import seaborn as sns
from streamlit_option_menu import option_menu
from utils import *
import plotly.graph_objects as go
import threading
from scipy.stats import weibull_min, expon, norm, gamma
st.set_page_config(page_title="PRODynamics", page_icon="⚙️", layout="wide")
#st.image('surviz_black_long.png')


 
# if selected == "Simulation & Results":
#     with st.spinner(text='In progress'):
#         time.sleep(3)
#         st.success('Done')
# if selected == "Contact":
#     st.subheader(f"**You Have selected {selected}**")


class PRODynamicsApp:
    def __init__(self):

        if 'line_data' not in st.session_state:  
            st.session_state.line_data = pd.read_excel("./LineData.xlsx", sheet_name="Line Data")
        if 'multi_ref_data' not in st.session_state:  
            st.session_state.multi_ref_data =  pd.read_excel("./LineData.xlsx", sheet_name="Multi-Ref")
        if "configuration" not in st.session_state:
            st.session_state.configuration = {}


        self.all_prepared = False
        self.selected = None
    

    def global_configuration(self):

        tab1, tab2 = st.tabs(["Simulation Data", "Stock Configuration"])
        st.session_state.configuration = {
        "sim_time": None,
        "takt_time": None,
        "n_robots": None,
        "strategy": None,
        "reset_shift": None,
        "stock_capacity": None,
        "initial_stock": None,
        "refill_time": None,
        "safety_stock": None,
        "refill_size": None,
        "n_repairmen": None,
        "enable_random_seed": None,
        "enable_breakdowns": None,
        "breakdown_dist_distribution": None
    }
    
        with tab1:
            columns = st.columns(2)
            # Machine Configuration
            with columns[0]:
                st.header("Simulation Data")
                st.session_state.configuration["sim_time"] = st.text_input("Simulation Time (s)", value="3600*24*7")
                st.session_state.configuration["takt_time"] = st.text_input("Expected Takt Time", value="10000")
                st.session_state.configuration["n_robots"] = st.number_input("Number of Handling Resources (Robots)", value=1)
                st.session_state.configuration["strategy"] = st.selectbox("Robot's Load/Unload Strategy", ["Balanced Strategy", "Greedy Strategy"])
                st.session_state.configuration["reset_shift"] = st.checkbox("Enable Shift Reseting", value=False)

            with columns[1]:
                # Breakdowns Configuration
                st.header("Breakdowns Configuration")
                st.session_state.configuration["enable_breakdowns"] = st.checkbox("Enable Machines Breakdown", value=True)
                st.session_state.configuration["n_repairmen"] = st.number_input("Number of Repairmen", value=3)
                st.session_state.configuration["enable_random_seed"] = st.checkbox("Enable Random Seed", value=True)
                st.session_state.configuration["breakdown_dist_distribution"] = st.selectbox("Probability Distribution", ["Weibull Distribution", "Exponential Distribution", "Normal Distribution", "Gamma Distribution"])

        # Stock Configuration
        with tab2:
            columns = st.columns(2)
            with columns[0]:
                st.header("Stock Configuration")
                st.session_state.configuration["stock_capacity"] = st.text_input("Input Stock Capacity", value="1")
                st.session_state.configuration["initial_stock"] = st.text_input("Initial Input Stock", value="0")
                st.session_state.configuration["refill_time"] = st.text_input("Refill Time (s)", value="To be chosen later per product.", disabled=True)
                st.session_state.configuration["safety_stock"] = st.text_input("Safety Stock", value="20")
                st.session_state.configuration["refill_size"] = st.text_input("Refill Size", value="1")
            
           

        if st.button("Confirm"):
            
            #task_assignement = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3,  3, 3, 3, 3, 3, 3, 3 ]
            self.save_global_settings()
            st.markdown("Simulation prepared")

            

    def home(self):
        row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((.1, 2.3, .1, 1.3, .1))
        with row0_1:
            st.title('PRODynamics')
        with row0_2:
            st.text("")
            st.subheader('@FORVIA')
            
        row3_spacer1, row3_1, row3_spacer2 = st.columns((.1, 3.2, .1))
        with row3_1:
            st.markdown("PRODynamics is an all-in-one solution for streamlining the evaluation and optimization of production lines. From configuration to simulation and optimization, we've got you covered.")
            st.markdown("The evaluation of production line performance within PRODynamics is grounded in a comprehensive understanding of the  dynamics and stochastic behaviors inherent in manufacturing operationss, ranging from machine breakdowns and delays to micro-stops and resource constraints.")

            st.markdown("This work is a culmination of an Idustrial PhD Project (CIFRE) conducted by [Anass ELHOUD](https://elhoud.me), aimed at harnessing the power of digital technologies and artificial intelligence to expedite the design and optimization of manufacturing process lines. ")
            
        # Create a row layout
        st.markdown('## Documentation')
        c1, c2= st.columns(2)
        c3, c4= st.columns(2)



        with st.container():
            c1.subheader("Weibull Distribution")
            c1.write("This distribution is versatile and exhibits a range of failure characteristics, including increasing, decreasing, or constant failure rates.")

            c2.subheader("Exponential Distribution")
            c2.write("This is a memoryless distribution: the probability of failure in the next time interval remains constant regardless of how much time has passed.")


        with st.container():
            c3.subheader("Normal Distribution")
            c3.write("The Normal distribution generates failure events in a symmertrically distributed way around MTTF.")

            c4.subheader("Gamma Distribution")
            c4.write("The Gamma distribution is memory-based, the probability of failure in the next time interval changes regarding of how much time has passed.")

        with c1:
            c11, c12= st.columns(2)
            with c11:
                # MTTF (Mean Time To Failure) in hours
                MTTF =  st.slider('MTTF (days)', min_value=0.0, max_value=30.0, value=1.0, step=1.0, key="mttf_weibull")
            with c12:
                beta = st.slider('Beta Parameter', min_value=0.0, max_value=30.0, value=1.0, step=1.0, key="beta_weibull")
            
            eta = MTTF
            x_weib = np.linspace(0, 3*MTTF, 1000) 
            pdf_weib = weibull_min.pdf(x_weib, beta, scale=eta)

            # Visualize Weibull distribution function using line chart 
            data_weib = pd.DataFrame({'Days': x_weib, 'Probability': pdf_weib})
            st.line_chart(data_weib, x="Days", y="Probability")

            
        with c2:
            MTTF = st.slider('MTTF (days)', min_value=0.0, max_value=30.0, value=1.0, step=1.0, key="tau_expo")
            
            lambda_param = 1 / MTTF
            x_exp = np.linspace(0, 3*MTTF, 1000) # Compute Weibull distribution function 
            
            exp_pdf =expon.pdf(x_exp, scale=lambda_param, loc=MTTF)
            data_exp = pd.DataFrame({'Days': x_exp, 'Probability': exp_pdf})
            st.line_chart(data_exp, x="Days", y="Probability")
            #st.hist(random_values, bins=50, density=True, alpha=0.6, color='g')

        with c3:
            c31, c32= st.columns(2)
            with c31:
                # MTTF (Mean Time To Failure) in hours
                MTTF =  st.slider('MTTF (days)', min_value=0.0, max_value=30.0, value=1.0, step=1.0, key="mttf_normal")
            with c32:
                dev = st.slider('Deviation', min_value=0.0, max_value=30.0, value=1.0, step=1.0, key="std_dev_normal")

            if dev == 0:
                std_dev = 0
            else:
                std_dev = MTTF / dev
            norm_x = np.linspace(0, 3*MTTF, 1000) 
            norm_pdf = norm.pdf(norm_x, loc=MTTF, scale=std_dev)
            data_norm = pd.DataFrame({'Days': norm_x, 'Probability': norm_pdf})
            st.line_chart(data_norm, x="Days", y="Probability")
                

        with c4:
            c41, c42= st.columns(2)
            with c41:
                # MTTF (Mean Time To Failure) in hours
                MTTF =  st.slider('MTTF (days)', min_value=0.0, max_value=30.0, value=1.0, step=1.0, key="mttf_gamma")
            with c42:
                k = st.slider('Lifespan', min_value=0.0, max_value=30.0, value=1.0, step=1.0, key="k_gamma")

            #theta = MTTF  # Scale parameter (mean time between breakdowns)

            # Generate values for the Gamma distribution
            gamma_x = np.linspace(0, 3*MTTF, 1000)  # Assuming a range of 3 times the mean time between breakdowns
            gamma_pdf = gamma.pdf(gamma_x, k, scale=MTTF)
            data_gamma = pd.DataFrame({'Days': gamma_x, 'Probability': gamma_pdf})
            st.line_chart(data_gamma, x="Days", y="Probability")

    def process_data(self):
        uploaded_file_line_data = st.file_uploader("Upload Multi-Reference Data", type=["xlsx", "xls", "csv"])
        tab1, tab2 = st.tabs(["Production Line Data", "Product Reference Data"])
        with tab1:
            #uploaded_file_line_data = st.file_uploader("Upload Production Line Data", type=["xlsx", "xls"])
            st.subheader("Production Line Data")
            if hasattr(st.session_state, 'line_data') and  isinstance(st.session_state.line_data, pd.DataFrame):
                updated_df = st.data_editor(st.session_state.line_data, num_rows="dynamic", key="data_edit")
                if not st.session_state.line_data.equals(updated_df):
                    st.session_state.line_data = updated_df.copy()
            else:
                with st.spinner('Uploading in progress...'):        
                    if uploaded_file_line_data is not None:
                        # st.session_state.line_data = pd.read_excel(uploaded_file_line_data, sheet_name="Line Data")
                        # st.data_editor(st.session_state.line_data, num_rows="dynamic", key="data_editor")

                        if uploaded_file_line_data.name.endswith('.csv'):
                            st.session_state.line_data  = pd.read_csv(uploaded_file_line_data)
                        elif uploaded_file_line_data.name.endswith(('.xls', '.xlsx')):
                            st.session_state.line_data  = pd.read_excel(uploaded_file_line_data, sheet_name="Line Data")
                            st.session_state.multi_ref_data = pd.read_excel(uploaded_file_line_data, sheet_name="Multi-Ref")
                        else:
                            st.error("Unsupported file format. Please upload a CSV or Excel file.")
                        st.data_editor(st.session_state.line_data, num_rows="dynamic", key="data_editor")


                with st.spinner('Uploading  in progress...'):        
                    if uploaded_file_line_data is not None:
                        # st.session_state.line_data = pd.read_excel(uploaded_file_line_data, sheet_name="Line Data")
                        # st.data_editor(st.session_state.line_data, num_rows="dynamic", key="data_editor")

                        if uploaded_file_line_data.name.endswith('.csv'):
                            st.session_state.line_data  = pd.read_csv(uploaded_file_line_data)
                        elif uploaded_file_line_data.name.endswith(('.xls', '.xlsx')):
                            st.session_state.line_data  = pd.read_excel(uploaded_file_line_data, sheet_name="Line Data")
                            st.session_state.multi_ref_data = pd.read_excel(uploaded_file_line_data, sheet_name="Multi-Ref")
                        else:
                            st.error("Unsupported file format. Please upload a CSV or Excel file.")
                        st.data_editor(st.session_state.line_data, num_rows="dynamic", key="data_editor")


        with tab2:
            st.subheader("Product Reference Data")
            if hasattr(st.session_state, 'multi_ref_data') and  isinstance(st.session_state.multi_ref_data, pd.DataFrame):
                updated_refs = st.data_editor(st.session_state.multi_ref_data, num_rows="dynamic",key="data_ref_edit")
                if not st.session_state.multi_ref_data.equals(updated_refs):
                    st.session_state.multi_ref_data = updated_refs.copy()
            else:
                
                if uploaded_file_line_data is not None:
                    if uploaded_file_line_data.name.endswith('.csv'):
                        st.session_state.multi_ref_data = pd.read_csv(uploaded_file_line_data)
                    elif uploaded_file_line_data.name.endswith(('.xls', '.xlsx')):
                        st.session_state.multi_ref_data = pd.read_excel(uploaded_file_line_data, sheet_name="Multi-Ref")
                    else:
                        st.error("Unsupported file format. Please upload a CSV or Excel file.")
                    # st.session_state.multi_ref_data = pd.read_excel(uploaded_file_line_data, sheet_name="Multi-Ref")
                    # st.subheader("Multi-Reference Data")
                    st.data_editor(st.session_state.multi_ref_data, num_rows="dynamic")

            new_ref_name = st.text_input("Enter new reference name")
            if st.button("+ Add Reference", key="add_new_ref"):
                if hasattr(st.session_state, 'multi_ref_data') and isinstance(st.session_state.multi_ref_data, pd.DataFrame):
                    if new_ref_name:
                        st.session_state.multi_ref_data[new_ref_name] = ""
                        st.rerun()
                    else:
                        st.warning("Please enter a column name")
                    st.subheader("Product Reference Data")
                    st.data_editor(st.session_state.multi_ref_data, num_rows="dynamic")


    def simulation_page(self):

        st.markdown("##### Simulation Data Summary")

        column1, column2, column3, column4 = st.columns(4)

        # Print the values of the variables in each column
        with column1:
            st.write("Simulation Time (s):", format_time(eval(st.session_state.configuration["sim_time"])))
            st.write("Expected Takt Time:", st.session_state.configuration["takt_time"])
            st.write("Number of Machines : ", len(st.session_state.line_data.values.tolist()))

        with column3:
            st.write("Machines Breakdown:", st.session_state.configuration["enable_breakdowns"])
            st.write("Probability Law: ", st.session_state.configuration["breakdown_dist_distribution"].replace(" Distribution", ""))
            st.write("Number of Repairmen:", st.session_state.configuration["n_repairmen"])
            
        with column2:
            st.write("Number of Handlers:", st.session_state.configuration["n_robots"])
            st.write("Handling Strategy:", st.session_state.configuration["strategy"].replace(" Strategy", ""))
        
        with column4:
            st.write("Number of References : ", len(st.session_state.multi_ref_data.set_index('Machine').to_dict(orient='list')))
            st.write("Shift Reset:", st.session_state.configuration["reset_shift"])
            st.write("Enable Random Seed:", st.session_state.configuration["enable_random_seed"])
        
        st.markdown("""---""")

        c1, c2 = st.columns(2)
        if st.button("Run Simulation"):
            env = simpy.Environment()
            tasks = []
            config_file = 'config.yaml'
            self.manuf_line = ManufLine(env, tasks, config_file=config_file)
            self.save_global_settings(self.manuf_line)
        
            self.manuf_line.references_config = st.session_state.multi_ref_data.set_index('Machine').to_dict(orient='list')
            self.manuf_line.machine_config_data = st.session_state.line_data.values.tolist()
            self.manuf_line.create_machines(st.session_state.line_data.values.tolist())
        
            self.all_prepared = True
            self.run_simulation(self.manuf_line)
            # simulation_thread = threading.Thread(target=self.run_simulation, args=(self.manuf_line,))
            # simulation_thread.start()



    def run_simulation(self, manuf_line):
        with st.spinner('Simulation in progress...'):
            manuf_line.run()
        st.success('Simulation completed!')

        st.markdown("""---""")
        st.header("Key Performance Indicators")

        # Global Cycle Time

        col = st.columns(5, gap='medium')
    
        with col[0]:
            global_cycle_time= manuf_line.sim_time/manuf_line.shop_stock_out.level
            delta_target = (float(st.session_state.configuration["takt_time"])-global_cycle_time)/float(st.session_state.configuration["takt_time"])
            st.metric(label="# Simulated Production", value=format_time(manuf_line.sim_time))

        with col[1]:
            global_cycle_time= manuf_line.sim_time/manuf_line.shop_stock_out.level
            delta_target = (float(st.session_state.configuration["takt_time"])-global_cycle_time)/float(st.session_state.configuration["takt_time"])
            st.metric(label="# Global Cycle Time", value=str(int(global_cycle_time))  +" s", delta=f"{delta_target:.2%}")
        
        with col[2]:
            global_cycle_time= manuf_line.sim_time/manuf_line.shop_stock_out.level
            st.metric(label="# Efficiency Rate", value=int(global_cycle_time), delta=str((float(st.session_state.configuration["takt_time"])-global_cycle_time)/float(st.session_state.configuration["takt_time"]))+" %")
        
        with col[3]:
            global_cycle_time= manuf_line.sim_time/manuf_line.shop_stock_out.level
            st.metric(label="# Efficiency Rate", value=int(global_cycle_time), delta=str((float(st.session_state.configuration["takt_time"])-global_cycle_time)/float(st.session_state.configuration["takt_time"]))+" %")

        with col[4]:
            oee =manuf_line.takt_time/global_cycle_time
            st.metric(label="# OEE / TRS", value=f"{oee:.2%}")

        machines_names = [m.ID for m in manuf_line.list_machines]
        idle_times = []
        machines_CT = []
        idle_times_sum = []
        
        for i, machine in enumerate(manuf_line.list_machines):
            idle_times_machine = []
            
            #for entry, exit in zip(machine.entry_times, machine.exit_times):
            ct_machine = []
            for finished in machine.finished_times:
                if finished is None:
                    ct_machine.append(0)
                else:
                    ct_machine.append(finished)
            machines_CT.append(np.sum(ct_machine))

            for time in machine.exit_times:
                idle_times_machine.append(time)
            
            idle_times.append(np.mean(idle_times_machine))
            idle_times_sum.append(np.sum(idle_times_machine)/manuf_line.sim_time)

        # Display plots
        st.header("Plots")
        c11, c12, c13= st.columns([0.5,0.3,0.2])
        c3, c4= st.columns([0.5,0.5])
        with c11:
            # st.subheader("Additional Plot 1")
            # chart_data = pd.DataFrame(np.random.randn(20, 3), columns=['a', 'b', 'c'])
            # st.bar_chart(chart_data)
            cycle_times = [t[0] / t[1] for t in manuf_line.output_tracks if t[1] !=0]

            # Create a list of time points for the x-axis
            time_points = list(range(len(cycle_times)))

            # Create a Plotly figure
            fig = go.Figure()
            for i, machine in enumerate(manuf_line.list_machines):
                idle_times_machine = []
                
                #for entry, exit in zip(machine.entry_times, machine.exit_times):
                ct_machine = []
                for finished in machine.finished_times:
                    if finished is None:
                        ct_machine.append(0)
                    else:
                        ct_machine.append(finished)
                machines_CT.append(ct_machine)
                #fig.add_trace(go.Scatter(x=list(range(len(machines_CT[-1]))), y=machines_CT[-1], mode='lines', name=machine.ID))
                #fig.add_trace(go.Scatter(x=[t[0] for t in manuf_line.machines_output[i]], y=[t[1] for t in manuf_line.machines_output[i]], mode='lines', name=machine.ID))


            # Add a line trace for the cycle time
            #x=time_points, y=cycle_times
            fig.add_trace(go.Scatter(x=time_points, y=cycle_times, mode='lines', name='Cycle Time', marker_color='blue'))

            # Update layout
            fig.update_layout(
                title='Evolution of Cycle Time',
                xaxis_title='Time',
                yaxis_title='Cycle Time',
                margin=dict(l=0, r=0, t=30, b=20)
            )

            # Display the Plotly figure
            st.plotly_chart(fig, use_container_width=True)



        with c12:

            #st.subheader("Additional Plot 2")
            # chart_data = pd.DataFrame(np.random.randn(20, 3), columns=['a', 'b', 'c'])
            # st.area_chart(chart_data)
            fig3 = go.Figure()
            buffer_out = [t[1] for t in manuf_line.output_tracks]
            times = [t[0] for t in manuf_line.output_tracks]
            fig3.add_trace(go.Scatter(x=times, y=buffer_out, mode='lines'))


            # Add a line trace for the cycle time
            #x=time_points, y=cycle_times
            #fig.add_trace(go.Scatter(x=time_points, y=cycle_times, mode='lines', name='Cycle Time', marker_color='blue'))

            # Update layout
            fig3.update_layout(
                title='Evolution of Production Rate',
                xaxis_title='Time',
                yaxis_title='Produced Parts',
                margin=dict(l=0, r=0, t=30, b=20)
            )

            # Display the Plotly figure
            st.plotly_chart(fig3, use_container_width=True)

        with c13:
            items_per_reference = []
            for item in list(manuf_line.references_config.keys()):
                items_per_reference.append(manuf_line.inventory_out.items.count(item))

            # Convert the dictionary to lists for Plotly
            reference_names = list(manuf_line.references_config.keys())
            item_counts = items_per_reference

  
            # Create the Plotly figure
            fig = go.Figure()

            # Add a bar trace for the number of items per reference
            fig.add_trace(go.Bar(x=reference_names, y=item_counts, marker_color="#77B4B4"))

            # Update layout
            fig.update_layout(
                title="Number of Products",
                xaxis_title="Reference",
                yaxis_title="Number of Items",
                xaxis=dict(type='category'),  # Specify category type for x-axis
                margin=dict(l=0, r=0, t=30, b=20)
            )

            # Display the Plotly figure
            st.plotly_chart(fig, use_container_width=True)

        with c3:
            fig_m, ax_m = plt.subplots()
            ax_m.set_ylabel('Percentage (%)')
            # ax_m.set_title('Machine Utilization Rate')
            
            # Calculate machine efficiency rate, machine available percentage, breakdown percentage, and waiting time percentage
            available_machines = []
            for m in manuf_line.list_machines:
                print("List of produced per machine = ", [m.ref_produced.count(ref) for ref in  manuf_line.references_config.keys() ])
                available_machine = np.sum([float(manuf_line.references_config[ref][manuf_line.list_machines.index(m)+1])* m.ref_produced.count(ref)  for ref in  manuf_line.references_config.keys()])/ manuf_line.sim_time
                #print("List available per machine = ", [float(manuf_line.references_config[ref][manuf_line.list_machines.index(m)+1])* m.ref_produced.count(ref)  for ref in  manuf_line.references_config.keys()])
                available_machines.append(100*available_machine)           
            #machine_available_percentage = [100*float(manuf_line.references_config[ref][manuf_line.list_machines.index(m)+1]) * m.ref_produced.count(ref) / manuf_line.sim_time for ref in  manuf_line.references_config.keys() for m in manuf_line.list_machines]
            machine_available_percentage = available_machines
            machine_available_percentage2 = [100 * ct / manuf_line.sim_time for m, ct in zip(manuf_line.list_machines, machines_CT)]
            #breakdown_percentage = [100 * float(m.MTTR * float(m.n_breakdowns)) / manuf_line.sim_time for m in manuf_line.list_machines]
            print("Nmb of breakdowns = ", [m.n_breakdowns for m in manuf_line.list_machines])
            print("Time of breakdown = ", [np.sum(m.real_repair_time) for m in manuf_line.list_machines])
            print("Time of breakdown = ", [np.sum(m.real_repair_time) for m in manuf_line.list_machines])
            
            breakdown_percentage = [100 * float(np.sum(m.real_repair_time)) / manuf_line.sim_time for m in manuf_line.list_machines]
            print('Breakdowns = ', breakdown_percentage)
            waiting_time_percentage = [100 - available_percentage - breakdown_percentage for available_percentage, breakdown_percentage in zip(machine_available_percentage, breakdown_percentage)]

            chart_data = {
                "Machine": machines_names,
                "Operating": machine_available_percentage,
                "Breakdown": breakdown_percentage,
                "Waiting": waiting_time_percentage,
            }
            
            
            colors = {
            "Operating": "green",
            "Breakdown": "red",
            "Waiting": "orange"
            }

            # Convert to DataFrame
            fig = go.Figure()

            # Add bar traces for each utilization type
            fig.add_trace(go.Bar(x=chart_data["Machine"], y=chart_data["Operating"], name="Operating", marker_color="green"))
            fig.add_trace(go.Bar(x=chart_data["Machine"], y=chart_data["Breakdown"], name="Breakdown", marker_color="red"))
            fig.add_trace(go.Bar(x=chart_data["Machine"], y=chart_data["Waiting"], name="Waiting", marker_color="orange"))

            # Update layout
            fig.update_layout(
                title="Machine Utilization Rate",
                xaxis_title="Machine",
                yaxis_title="Percentage (%)",
                barmode="stack",  # Stack bars on top of each other
                legend=dict(
                orientation="h",  # Horizontal legend
                xanchor="center",  # Anchor legend to the right
                x=0.5  # Adjust horizontal position of the legend
            ),
            margin=dict(l=0, r=0, t=30, b=30)
            )


            # Display the Plotly figure
   
            st.plotly_chart(fig, use_container_width=True)
            # # Plotting
            # bars1 = ax_m.bar(machines_names, machine_available_percentage, label='Operating', color="green")
            # bars2 = ax_m.bar(machines_names, breakdown_percentage, bottom=machine_available_percentage, label='Breakdown', color="red")
            # bars3 = ax_m.bar(machines_names, waiting_time_percentage, bottom=np.array(machine_available_percentage) + np.array(breakdown_percentage), label='Waiting', color="Orange")
            # ax_m.plot(machines_names, machine_efficiency_rate, '--x', color='white', label="Efficiency")
            # ax_m.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=4)

            # # Display the plot
            # st.pyplot(fig_m)

        # Plot of Machine Breakdowns
        with c4:
            breakdown_values = [m.n_breakdowns for m in manuf_line.list_machines]
            starvation_times = [m.waiting_time[0] for m in manuf_line.list_machines]
            blockage_times = [m.waiting_time[1] for m in manuf_line.list_machines]
            index = range(len(manuf_line.list_machines))

            fig2 = go.Figure()

            # Create traces for starvation time and blockage time

            fig2.add_trace(go.Bar(x=chart_data["Machine"], y=starvation_times, name='Starvation Time', marker_color="green"))
            fig2.add_trace(go.Bar(x=chart_data["Machine"], y=blockage_times, name='Blockage Time', marker_color="red"))

            fig2.update_layout(
                title="Starvation and Blockage Time per Machine",
                xaxis_title="Machine",
                yaxis_title="Waiting Time (s)",
                barmode="group",  # Stack bars on top of each other
                legend=dict(
                orientation="h",  # Horizontal legend
                xanchor="center",  # Anchor legend to the right
                x=0.5  # Adjust horizontal position of the legend
            ),
            margin=dict(l=0, r=0, t=30, b=30)
            )

            st.plotly_chart(fig2)

        # Additional Plots
       
        

    def run_main(self):
        #st.set_page_config(page_title="PRODynamics", page_icon="⚙️", layout="wide")

        with st.sidebar:
            selected = option_menu(
            menu_title = "Prodynamics",
            options = ["Home","Global Settings","Process Data","Simulation Lab","Optimization Lab","Contact Us"],
            icons = ["house","gear","activity","play-circle-fill","bar-chart-line-fill", "question-circle-fill"],
            menu_icon = "cast",
            default_index = 0,
            #orientation = "horizontal",
        )
            


        if selected == "Home":
            self.home()
        elif selected == "Global Settings":
            self.global_configuration()
        elif selected == "Process Data":
            self.process_data()
        elif selected == "Simulation Lab":
            self.simulation_page()
        elif selected == "Optimization Lab":
            # Add the content for the "Storage" section here
            pass
        elif selected == "Contact Us":
            # Add the content for the "Contact Us" section here
            pass


        
    def save_global_settings(self, manuf_line):
        configuration = st.session_state.configuration

        if configuration["enable_breakdowns"]:
            manuf_line.breakdowns_switch = True
        else:
            manuf_line.breakdowns_switch = False

        if configuration["enable_random_seed"]:
            manuf_line.randomseed = True
        else:
            manuf_line.randomseed = False

        available_strategies = ["Balanced Strategy", "Greedy Strategy"]

        manuf_line.stock_capacity = float(configuration["stock_capacity"])
        manuf_line.stock_initial = float(configuration["initial_stock"])
        manuf_line.reset_shift_dec = bool(configuration["reset_shift"])
        manuf_line.breakdown_law = str(configuration["breakdown_dist_distribution"])
        # if value1-value2, then the refill time is random between two values
        # pattern = r'^(\d+)-(\d+)$'
        # match = re.match(pattern, str(configuration["refill_time"]))
        # if match:
        #     value1 = int(match.group(1))
        #     value2 = int(match.group(2))
        #     manuf_line.refill_time = [value1, value2]
        #     print("refill time 1 = ", manuf_line.refill_time)
        # else:
        #     manuf_line.refill_time = float(configuration["refill_time"])
        
        manuf_line.safety_stock = float(configuration["safety_stock"])
        manuf_line.refill_size = float(configuration["refill_size"])
        manuf_line.n_robots = float(configuration["n_robots"])
        manuf_line.n_repairmen = int(configuration["n_repairmen"])
        manuf_line.robot_strategy = int(available_strategies.index(configuration["strategy"]))
        manuf_line.repairmen = simpy.PreemptiveResource(manuf_line.env, capacity=int(configuration["n_repairmen"]))
        manuf_line.supermarket_in = simpy.Container(manuf_line.env, capacity=manuf_line.stock_capacity, init=manuf_line.stock_initial)
        manuf_line.shop_stock_out = simpy.Container(manuf_line.env, capacity=float(manuf_line.config["shopstock"]["capacity"]), init=float(manuf_line.config["shopstock"]["initial"]))
        
        manuf_line.sim_time = eval(str(configuration["sim_time"]))
        print("sim time first = ",  manuf_line.sim_time)
        manuf_line.takt_time = eval(str(configuration["takt_time"]))
  


        

if __name__ == "__main__":
    app = PRODynamicsApp()
    app.run_main()