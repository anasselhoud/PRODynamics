import graphviz
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
from utils_optim import *
from utils_static import *
import plotly.graph_objects as go
import threading
from scipy.stats import weibull_min, expon, norm, gamma
from joblib import Parallel, delayed
from scipy.optimize import minimize
st.set_page_config(page_title="PRODynamics", page_icon="⚙️", layout="wide")
#st.image('surviz_black_long.png')




class PRODynamicsApp:
    def __init__(self):

        if 'line_data' not in st.session_state:  
            st.session_state.line_data = pd.read_excel("./LineData.xlsx", sheet_name="Line Data")
        if 'multi_ref_data' not in st.session_state:  
            st.session_state.multi_ref_data =  pd.read_excel("./LineData.xlsx", sheet_name="Multi-Ref")
        if "configuration" not in st.session_state:
            st.session_state.configuration = {
                "sim_time": "3600*24*7",
                "takt_time": "10000",
                "enable_robots": True,
                "strategy": "Balanced Strategy",
                "reset_shift": False,
                "dev_mode":False,
                "stock_capacity": "1",
                "safety_stock": "1",
                "n_repairmen": 3,
                "enable_random_seed": True,
                "enable_breakdowns": True,
                "breakdown_dist_distribution": "Weibull Distribution",
                "central_storage_enable": False,
                "central_storage_ttr": {'front': 100, "back": 100},
            }

        if "uploaded_file" not in st.session_state:
            st.session_state.uploaded_file = None

        if 'mbom_data' not in st.session_state:
            st.session_state.mbom_data = pd.read_xml('./assets/inputs/L76 Dual Passive MBOM.xml', xpath=".//weldings//welding")
            st.session_state.parts_data = pd.read_xml('./assets/inputs/L76 Dual Passive MBOM.xml', xpath=".//parts//part")

        if "configuration_static" not in st.session_state:
            st.session_state.configuration_static = {
                "Exploration Mode": "Standard",
                "Search Speed": "Fast",
                "Target CT": "100",
                "Tolerance": "0.1",
            }
            
        if "static_optim_config" not in st.session_state:
            st.session_state.static_optim_config = {
                "takt_time": "100",
                "n_machines": 4,
                "n_stations_permachine": 2,
                "n_operators": 3,
                "ressource_list": [],
                "manu_op_assignments": {}
            }
        # Central Storage
        if "central_storage" not in st.session_state:        
            st.session_state.central_storage = {
                'front': pd.DataFrame([
                    {'Allowed references' : "",
                     'Capacity': 336}]),

                'back': pd.DataFrame([
                    {'Allowed references' : "",
                     'Capacity': 376}])
                }

        self.all_prepared = False
        self.selected = None

        if "static_best_solution" not in st.session_state: 
            st.session_state.static_best_solution = []

        if "detail_btn_run" not in st.session_state: 
            st.session_state.detail_btn_run = False

    def click_detail_static_optim(self):
        st.session_state.detail_btn_run = True


    def global_configuration(self):

        tab1, tab2 = st.tabs(["Simulation Data", "Stock Configuration"])
    
        with tab1:
            columns = st.columns(2)
            with columns[0]:
                st.header("Simulation Data")
                st.session_state.configuration["sim_time"] = st.text_input("Simulation Time (s)", value=st.session_state.configuration.get("sim_time", "3600*24*7"))
                st.session_state.configuration["takt_time"] = st.text_input("Expected Takt Time", value=st.session_state.configuration.get("takt_time", "10000"))
                st.session_state.configuration["strategy"] = st.selectbox("Load/Unload Strategy", ["Balanced Strategy", "Greedy Strategy"], index=0 if st.session_state.configuration.get("strategy") == "Balanced Strategy" else 1)
                st.session_state.configuration["reset_shift"] = st.checkbox("Enable Shift Reseting", value=st.session_state.configuration.get("reset_shift", False))
                st.session_state.configuration["dev_mode"] = st.checkbox("Enable developer mode (slow down !)", value=st.session_state.configuration.get("dev_mode", True))

            with columns[1]:
                st.header("Breakdowns Configuration")
                st.session_state.configuration["enable_breakdowns"] = st.checkbox("Enable Machines Breakdown", value=st.session_state.configuration.get("enable_breakdowns", True))
                st.session_state.configuration["breakdown_dist_distribution"] = st.selectbox("Probability Distribution", ["Weibull Distribution", "Exponential Distribution", "Normal Distribution", "Gamma Distribution"], index=["Weibull Distribution", "Exponential Distribution", "Normal Distribution", "Gamma Distribution"].index(st.session_state.configuration.get("breakdown_dist_distribution", "Weibull Distribution")))
                st.session_state.configuration["n_repairmen"] = st.number_input("Number of Repairmen", value=st.session_state.configuration.get("n_repairmen", 3))
                st.session_state.configuration["enable_random_seed"] = st.checkbox("Enable Random Seed", value=st.session_state.configuration.get("enable_random_seed", True))
                if st.session_state.configuration["breakdown_dist_distribution"]:
                    with st.expander("Customize the breakdown distribution"):
                        col1style, col2style = st.columns([1,1])
                        if st.session_state.configuration["breakdown_dist_distribution"] == "Weibull Distribution":
                            lifespan = col1style.number_input("Global Lifespan (s)")
                            shape_param = col2style.number_input("Shape Parameter (k)", value=st.session_state.configuration.get("shape_param", 1.5))
                            scale_param = col2style.number_input("Scale Parameter (λ)", value=st.session_state.configuration.get("scale_param", 1000))
                            st.session_state.configuration["shape_param"] = shape_param
                            st.session_state.configuration["scale_param"] = scale_param
                        elif st.session_state.configuration["breakdown_dist_distribution"] == "Exponential Distribution":
                            scale_param = col2style.number_input("Scale Parameter (λ)", value=st.session_state.configuration.get("scale_param", 1000))
                            st.session_state.configuration["scale_param"] = scale_param
                        elif st.session_state.configuration["breakdown_dist_distribution"] == "Normal Distribution":
                            mean_param = col2style.number_input("Mean", value=st.session_state.configuration.get("mean_param", 500))
                            std_param = col2style.number_input("Standard Deviation", value=st.session_state.configuration.get("std_param", 100))
                            st.session_state.configuration["mean_param"] = mean_param
                            st.session_state.configuration["std_param"] = std_param
                        elif st.session_state.configuration["breakdown_dist_distribution"] == "Gamma Distribution":
                            shape_param = col2style.number_input("Shape Parameter (α)", value=st.session_state.configuration.get("shape_param", 2))
                            scale_param = col2style.number_input("Scale Parameter (β)", value=st.session_state.configuration.get("scale_param", 500))
                            st.session_state.configuration["shape_param"] = shape_param
                            st.session_state.configuration["scale_param"] = scale_param

        with tab2:
            columns = st.columns(2)
            with columns[0]:
                st.header("Stock Configuration")
                st.session_state.configuration["stock_capacity"] = st.text_input("Input Stock Capacity", value=st.session_state.configuration.get("stock_capacity", "1"))
                st.session_state.configuration["safety_stock"] = st.text_input("Safety Stock", value=st.session_state.configuration.get("safety_stock", "20"))

    def home(self):
        row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((.1, 2.3, .1, 1.3, .1))
        st.image('./assets/image_banner.png')
            
        row3_spacer1, row3_1, row3_spacer2 = st.columns((.1, 3.2, .1))
        with row3_1:
            st.markdown("PRODynamics is an all-in-one solution for streamlining the evaluation and optimization of production lines. From configuration to simulation and optimization, we've got you covered.")
            st.markdown("The evaluation of production line performance within PRODynamics is grounded in a comprehensive understanding of the  dynamics and stochastic behaviors inherent in manufacturing operationss, ranging from machine breakdowns and delays to micro-stops and resource constraints.")

            st.markdown("This work is a culmination of an Industrial PhD Project (CIFRE) conducted by [Anass ELHOUD](https://elhoud.me), aimed at harnessing the power of digital technologies and artificial intelligence to expedite the design and optimization of manufacturing process lines. ")
            
        # Create a row layout
        st.markdown('## Documentation')

        st.markdown("### UI Walkthrough")
        video, text= st.columns(2)
        with video:
            VIDEO_URL = "https://cdn.pixabay.com/video/2016/12/31/6962-197634410_large.mp4"
            st.video(VIDEO_URL)
        with text:
            st.markdown("#### What's New?")
            st.markdown("""
            ##### V0.1 - Pre-Release:
            * A new, more user-friendly and optimized interface
            * Improved simulation performance
            * Enhanced data visualization
            * Introduction of new optimization tools
            * Improved documentation
            * Bug fixes and performance enhancements
            """)
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
              # Excel file loader
        uploaded_file = st.file_uploader("Upload line & multi-reference data", type=["xlsx", "xls"])

        if (uploaded_file is not None) and (uploaded_file != st.session_state.uploaded_file) :
            with st.spinner('Uploading in progress...'): 

                if uploaded_file.name.endswith(('.xls', '.xlsx')):
                    st.session_state.uploaded_file = uploaded_file
                    excel_df = pd.read_excel(st.session_state.uploaded_file, sheet_name=["Line Data", "Multi-Ref"])
                    st.session_state.line_data  = excel_df["Line Data"].copy()
                    st.session_state.multi_ref_data = excel_df["Multi-Ref"].copy()

                else:
                    st.error("Unsupported file format. Please upload a CSV or Excel file.")
            st.rerun()  

        # Tabs
        tab1, tab2, tab3 = st.tabs(["Production Line Data", "Product Reference Data", "Central Storage"])
        with tab1:
            st.subheader("Production Line Data")

            st.session_state.configuration["enable_robots"] = st.toggle('Enable robots', value=st.session_state.configuration["enable_robots"], key='toggle_robots')

            if hasattr(st.session_state, 'line_data') and  isinstance(st.session_state.line_data, pd.DataFrame):
                line_data_editor = st.data_editor(st.session_state.line_data, num_rows="dynamic", key="line_data_edit")
            
            # Save button
            if st.button("Save Data", key="line_data_save"):
                st.session_state.line_data = line_data_editor.copy()
                st.rerun()

            # Graph display
            display_btn = st.button("Display the line")

            with st.columns([0.1, 0.8, 0.1])[1]:
                if display_btn:
                    graph = graphviz.Digraph()
                    graph.attr(rankdir='LR')

                    #TODO: find a way to check if the dark mode is activated (change main color based on that)
                    main_color = "white"

                    #graph.attr(bgcolor='black')
                    graph.attr(bgcolor='transparent')

                    # Legend for central storage
                    if st.session_state.configuration["central_storage_enable"]:
                        graph.node("Central Storage", shape='box', style='filled', fontcolor='dark', fillcolor='#77B5FE', height='0.3', width='0.5', margin='0.01')

                    # Generate nodes, and colour machines linked to the central storage
                    for machine, can_fill_cs in zip(st.session_state.line_data["Machine"].values, st.session_state.line_data["Fill central storage"].values):
                        if st.session_state.configuration["central_storage_enable"] and can_fill_cs:
                            graph.node(machine, style='filled', fillcolor='#77B5FE',color=main_color, fontcolor=main_color)
                        else:
                            graph.node(machine, color=main_color, fontcolor=main_color)

                    # Generate the edges origin -> destination
                    for origin, destination in zip(st.session_state.line_data["Machine"].values, st.session_state.line_data["Link"].values):
                        try:
                            destinations = eval(destination)
                            for dest in destinations:
                                graph.edge(origin, dest, color=main_color, fontcolor=main_color)
                        except:
                            graph.edge(origin, destination, color=main_color, fontcolor=main_color)

                    # Plot
                    graph.node("END", color=main_color, fontcolor=main_color)
                    st.graphviz_chart(graph, use_container_width=True)

        with tab2:
            st.subheader("Product Reference Data")
            uploaded_file_line_data_ref = st.file_uploader("Upload Product Reference Data", type=[ "csv", "xlsx", "xls"], key="upload_refs")
            if hasattr(st.session_state, 'multi_ref_data') and  isinstance(st.session_state.multi_ref_data, pd.DataFrame):
                ref_data_editor = st.data_editor(st.session_state.multi_ref_data, num_rows="dynamic", key="ref_data_edit")
                
            # Save button
            if st.button("Save", key="ref_data_save"):
                st.session_state.multi_ref_data = ref_data_editor.copy()
                st.rerun()


            new_ref_name = st.text_input("Add a new reference", placeholder="Reference name")
            if st.button("Add reference", key="add_new_ref"):
                if hasattr(st.session_state, 'multi_ref_data') and isinstance(st.session_state.multi_ref_data, pd.DataFrame):
                    if new_ref_name:
                        st.session_state.multi_ref_data[new_ref_name] = ""
                        st.rerun()
                    else:
                        st.warning("Please enter a column name")

        # Central Storage
        with tab3:
            st.subheader("Central Storage")                
            st.session_state.configuration["central_storage_enable"] = st.checkbox("Enable", value=st.session_state.configuration["central_storage_enable"])
                
            # Allow all references by default according 
            all_references = st.session_state.multi_ref_data.columns.to_list()[1:]

            sides = ['front', 'back']
            for side in sides:
                for i, value in enumerate(st.session_state.central_storage[side]['Allowed references'].values):
                    if not value:
                        st.session_state.central_storage[side]['Allowed references'].values[i] = str(all_references)

            # Display for each side of the central storage
            cols = st.columns(len(sides))
            for i, side in enumerate(sides):
                with cols[i]:
                    # Side, time to reach, table to edit date, save button
                    st.subheader(side.capitalize())
                    st.session_state.configuration['central_storage_ttr'][side] = st.number_input("Time to reach (s)", value=st.session_state.configuration['central_storage_ttr'][side], format='%i', key=f'central_storage_ttr_{side}')
                    editor = st.data_editor(st.session_state.central_storage[side], num_rows="dynamic", key=f"central_storage_{side}")

                    if st.button("Save", key=f"central_storage_save_{side}"):
                        st.session_state.central_storage[side] = editor.copy()

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
            st.write("Number of Robots:", len(list(set(st.session_state.line_data["Transporter ID"].to_list()))))
            st.write("Handling Strategy:", st.session_state.configuration["strategy"].replace(" Strategy", ""))
            st.write("Central Storage:", st.session_state.configuration["central_storage_enable"])
        
        with column4:
            st.write("Number of References : ", len(st.session_state.multi_ref_data.set_index('Machine').to_dict(orient='list')))
            st.write("Shift Reset:", st.session_state.configuration["reset_shift"])
            st.write("Enable Random Seed:", st.session_state.configuration["enable_random_seed"])
        
        # Display manufacturing line graph
        with st.columns([0.2, 0.6, 0.2])[1]:
            graph = graphviz.Digraph()
            graph.attr(rankdir='LR')

            #TODO: find a way to check if the dark mode is activated (change main color based on that)
            main_color = "white"

            #graph.attr(bgcolor='black')
            graph.attr(bgcolor='transparent')

            # Legend for central storage
            if st.session_state.configuration["central_storage_enable"]:
                graph.node("Central Storage", shape='box', style='filled', fontcolor='dark', fillcolor='#77B5FE', height='0.3', width='0.5', margin='0.01')

            # Generate nodes, and colour machines linked to the central storage
            for machine, can_fill_cs in zip(st.session_state.line_data["Machine"].values, st.session_state.line_data["Fill central storage"].values):
                if st.session_state.configuration["central_storage_enable"] and can_fill_cs:
                    graph.node(machine, style='filled', fillcolor='#77B5FE',color=main_color, fontcolor=main_color)
                else:
                    graph.node(machine, color=main_color, fontcolor=main_color)

            # Generate the edges origin -> destination
            for origin, destination in zip(st.session_state.line_data["Machine"].values, st.session_state.line_data["Link"].values):
                try:
                    destinations = eval(destination)
                    for dest in destinations:
                        graph.edge(origin, dest, color=main_color, fontcolor=main_color)
                except:
                    graph.edge(origin, destination, color=main_color, fontcolor=main_color)

            # Plot
            graph.node("END", color=main_color, fontcolor=main_color)
            st.graphviz_chart(graph, use_container_width=True)

        st.markdown("""---""")

        c1, c2 = st.columns(2)
        if st.button("Run Simulation"):
            self.all_prepared = True

            env = simpy.Environment()
            tasks = []

            configuration = st.session_state.configuration
            references_config = st.session_state.multi_ref_data.set_index('Machine').to_dict(orient='list')
            line_data = st.session_state.line_data.values.tolist()

            self.manuf_line = ManufLine(env, tasks, config_file='config.yaml')
            self.manuf_line.save_global_settings(configuration, references_config, line_data, buffer_sizes=[])
            self.manuf_line.create_machines(line_data)

            # Check robots data consistency  
            if st.session_state.configuration['enable_robots']:
                col_to_check = ["Transport Time", "Transport Order", "Transporter ID"]
                for col in col_to_check:
                    if any([False if isinstance(value, str) else np.isnan(value) for value in st.session_state.line_data[col].to_list()]):
                        st.error(f'You enabled robots but column "{col}" in "Process Data > Production Line Data" has an empty value. Please, either fill the value or disable robots before running again.', icon="🚨")
                        self.all_prepared = False
                        break

            # Set up the central storage
            if st.session_state.configuration["central_storage_enable"]:
                # Turn pd.DataFrame into dict to 
                central_storage_config = {}
                for side in ['front', 'back']:
                    central_storage_config[side] = []
                    for allowed, capacity in zip(st.session_state.central_storage[side]['Allowed references'].values, st.session_state.central_storage[side]['Capacity'].values):
                        central_storage_config[side].append({'allowed_ref': eval(allowed), 'capacity': capacity})

                self.manuf_line.central_storage = CentralStorage(env, self.manuf_line, central_storage_config, st.session_state.configuration["central_storage_ttr"])

            if self.all_prepared:
                self.run_simulation(self.manuf_line)
            # simulation_thread = threading.Thread(target=self.run_simulation, args=(self.manuf_line,))
            # simulation_thread.start()

    def optimization_page(self):

        st.markdown("##### Line Data Summary")

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
            st.write("Number of Robots:", len(list(set(st.session_state.line_data["Transporter ID"].to_list()))))
            st.write("Handling Strategy:", st.session_state.configuration["strategy"].replace(" Strategy", ""))
        
        with column4:
            st.write("Number of References : ", len(st.session_state.multi_ref_data.set_index('Machine').to_dict(orient='list')))
            st.write("Shift Reset:", st.session_state.configuration["reset_shift"])
            st.write("Enable Random Seed:", st.session_state.configuration["enable_random_seed"])
        
        st.markdown("""---""")

        c1, c2 = st.columns(2)
        with c1:
            inventory_cost = st.text_input("Inventory Unit Cost", value="10")
        with c2:
            part_unit_gain = st.text_input("Produced Unit Gain", value="100")


        if st.button("Run Optimization"):
            reference_config = st.session_state.multi_ref_data.set_index('Machine').to_dict(orient='list')
            line_data = st.session_state.line_data.values.tolist()
            configuration = st.session_state.configuration
            buffer_capacities = []
            
            num_buffers = len(line_data)
            step_size = 10
            max_capacity = 100

            
            progress_text = "Operation in progress. Please wait."
            my_bar = st.progress(0, text=progress_text)
            combinations = []

            for i in range(num_buffers):
                for capacity in range(1, max_capacity + 1, step_size):
                    current_combination = [1] * num_buffers  
                    current_combination[i] = capacity  
                    combinations.append(current_combination)

            costs = []
            raw_results = []
            _, waiting_ref, _, _= buffer_optim_costfunction([1 for _ in range(num_buffers)], configuration, reference_config, line_data)
            for i, candidate in enumerate(combinations):
                total_cost, raw_result = function_to_optimize(candidate, configuration, reference_config, line_data, waiting_ref, float(inventory_cost), float(part_unit_gain))
                raw_results.append(raw_result)
                costs.append(total_cost)
                my_bar.progress((i + 1) /len(combinations))
                csv_file_path = 'costs_per_candidate.csv'

                with open(csv_file_path, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Candidate', 'Cost'])
                    for candidate, cost in zip(combinations, raw_results):
                        writer.writerow([candidate, cost])

            fig = go.Figure()
            st.write("Optimal Storage Size:")
            optim_results = []
            coeffs_all = []
            def objective_function(bc, coeffs):
                return -coeffs[0]*bc**3 - coeffs[1]*bc**2 - coeffs[2]*bc - coeffs[3]
            
            cols = st.columns(3)
            for i in range(num_buffers):
                capacities = [x_buffers[i] for x_buffers in combinations][i*int(max_capacity/step_size):(i+1)*int(max_capacity/step_size)]
                results = [result[i] for result in costs][i*int(max_capacity/step_size):(i+1)*int(max_capacity/step_size)]
                x = np.array(capacities)
                y = np.array(results)
                coeffs = np.polyfit(x, y, 3)
                coeffs_all.append(coeffs)
                y_pred = [np.sum([coeffs[i]*x_ex**(len(coeffs)-i-1) for i in range(len(coeffs))]) for x_ex in x ]
                bounds = [(1, None)]
                result = minimize(objective_function, x0=1, args=(coeffs), bounds=bounds)
                optimum_x = result.x[0]
                if st.session_state.configuration["dev_mode"]:
                    print("Optimal = ", optimum_x)
                #fig.add_trace(go.Scatter(x=capacities, y=results))
                optim_results.append(optimum_x)
                fig.add_trace(go.Scatter(x=capacities, y=y_pred, mode='lines', name=f'Buffer {i + 1}'))
                with cols[i%3]:
                    st.write("Buffer " + str(i+1) + ' Size = ' + str(int(optim_results[i])))


            fig.update_layout(
                title='Evolution of Benefit per Buffer Size',
                xaxis_title='Buffer Storage Size',
                yaxis_title='Benefit (Cash Unit)',
                margin=dict(l=0, r=0, t=30, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

        
        # if st.button("Run Action"):
        #     reference_config = st.session_state.multi_ref_data.set_index('Machine').to_dict(orient='list')
        #     line_data = st.session_state.line_data.values.tolist()
        #     configuration = st.session_state.configuration
        #     env = simpy.Environment()
        #     tasks = []
        #     config_file = 'config.yaml'

        #     self.manuf_line = ManufLine(env, tasks, config_file=config_file)
        #     self.manuf_line.save_global_settings(self.manuf_line)
        
        #     self.manuf_line.create_machines(st.session_state.line_data.values.tolist())
        #     # self.manuf_line.initialize()

    
        #     actions = [
        #         [self.manuf_line.supermarket_in, self.manuf_line.list_machines[0]],
        #         [self.manuf_line.supermarket_in, self.manuf_line.list_machines[1]],
        #         [self.manuf_line.list_machines[0], self.manuf_line.list_machines[2]],
        #         [self.manuf_line.list_machines[0], self.manuf_line.list_machines[2]],
        #         [self.manuf_line.supermarket_in, self.manuf_line.list_machines[0]],
        #         [self.manuf_line.supermarket_in, self.manuf_line.list_machines[0]],
        #         [self.manuf_line.list_machines[2], self.manuf_line.list_machines[3]],
        #         [self.manuf_line.list_machines[0], self.manuf_line.list_machines[2]],
        #         [self.manuf_line.list_machines[3], self.manuf_line.list_machines[4]],]
            
        #     for action in actions:
        #         # Run each action
        #         self.manuf_line.run_action(action)
        #         for m in self.manuf_line.list_machines:
        #             print(m.ID + " - Operating = " +str(m.operating) + " - " + str(len(m.buffer_in.items)) + " | " + str(len(m.buffer_out.items)) + "   -- " + str(m.waiting_time))
        #             print("Level = ", len(self.manuf_line.shop_stock_out.items))

    def run_simulation(self, manuf_line):
        # Run simulation and visualize its progress using a progress bar
        my_bar_simulation = st.progress(0, text="Simulation in progress...")
        manuf_line.run(my_bar_simulation)
        my_bar_simulation.empty()
        st.success('Simulation completed!')

        st.markdown("""---""")

        st.header("Key Performance Indicators")

        # Global Cycle Time
        col = st.columns(5, gap='medium')
    
        with col[0]:
            if st.session_state.configuration["dev_mode"]:
                print("last machine level = ",manuf_line.list_machines[-1].last )
            global_cycle_time= manuf_line.sim_time/len(manuf_line.shop_stock_out.items)
            delta_target = (float(st.session_state.configuration["takt_time"])-global_cycle_time)/float(st.session_state.configuration["takt_time"])
            st.metric(label="# Simulated Production", value=format_time(manuf_line.sim_time))

        with col[1]:
            global_cycle_time= manuf_line.sim_time/len(manuf_line.shop_stock_out.items)
            delta_target = (float(st.session_state.configuration["takt_time"])-global_cycle_time)/float(st.session_state.configuration["takt_time"])
            st.metric(label="# Global Cycle Time", value=str(int(global_cycle_time))  +" s", delta=f"{delta_target:.2%}")
        
        with col[2]:
            global_cycle_time= manuf_line.sim_time/len(manuf_line.shop_stock_out.items)
            st.metric(label="# Efficiency Rate", value=int(global_cycle_time), delta=str((float(st.session_state.configuration["takt_time"])-global_cycle_time)/float(st.session_state.configuration["takt_time"]))+" %")
        
        with col[3]:
            global_cycle_time= manuf_line.sim_time/len(manuf_line.shop_stock_out.items)
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
        c3, c4, c5 = st.columns([0.4,0.4, 0.2])
        with c11:
            # st.subheader("Additional Plot 1")
            # chart_data = pd.DataFrame(np.random.randn(20, 3), columns=['a', 'b', 'c'])
            # st.bar_chart(chart_data)
            cycle_times = [t[0] / t[1] for t in manuf_line.output_tracks if t[1] !=0]
            cycle_times_per_ref = []
            
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
            fig.add_trace(go.Scatter(x=time_points, y=cycle_times, mode='lines', name='Global CT', marker_color='blue'))

            for ref_ind, ref in enumerate(manuf_line.references_config.keys()):
                cycle_times_per_ref = [t[0] / t[1] for t in manuf_line.output_tracks_per_ref[ref_ind] if t[1] !=0]
                fig.add_trace(go.Scatter(x=list(range(len(cycle_times_per_ref))), y=cycle_times_per_ref, mode='lines', name=ref))
            # Update layout
            fig.update_layout(
                title='Evolution of Cycle Time',
                xaxis_title='Time',
                yaxis_title='Cycle Time (s)',
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
                items_per_reference.append(manuf_line.shop_stock_out.items.count(item))

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
            if st.session_state.configuration["dev_mode"]:
                for op in manuf_line.manual_operators:
                    print("Operator WC = ", op.wc)

            # Calculate machine efficiency rate, machine available percentage, breakdown percentage, and waiting time percentage
            available_machines = []
            for m in manuf_line.list_machines:
                if st.session_state.configuration["dev_mode"]:
                    print(f"Produced by machine {m.ID} = ", [m.ref_produced.count(ref) for ref in  manuf_line.references_config.keys() ])
                    print("Waiting for op = ", np.mean(m.wc))

                available_machine = np.sum([float(manuf_line.references_config[ref][manuf_line.list_machines.index(m)+3])* m.ref_produced.count(ref)  for ref in  manuf_line.references_config.keys()])/ manuf_line.sim_time
                #print("List available per machine = ", [float(manuf_line.references_config[ref][manuf_line.list_machines.index(m)+3])* m.ref_produced.count(ref)  for ref in  manuf_line.references_config.keys()])
                available_machines.append(100*available_machine)           
            #machine_available_percentage = [100*float(manuf_line.references_config[ref][manuf_line.list_machines.index(m)+3]) * m.ref_produced.count(ref) / manuf_line.sim_time for ref in  manuf_line.references_config.keys() for m in manuf_line.list_machines]
            machine_available_percentage = available_machines
            machine_available_percentage2 = [100 * ct / manuf_line.sim_time for m, ct in zip(manuf_line.list_machines, machines_CT)]
            #breakdown_percentage = [100 * float(m.MTTR * float(m.n_breakdowns)) / manuf_line.sim_time for m in manuf_line.list_machines]
            if st.session_state.configuration["dev_mode"]:
                print("Nmb of breakdowns = ", [m.n_breakdowns for m in manuf_line.list_machines])
                print("Time of breakdown = ", [np.sum(m.real_repair_time) for m in manuf_line.list_machines])
                print("Time of breakdown = ", [np.sum(m.real_repair_time) for m in manuf_line.list_machines])
                print(manuf_line.central_storage)
            breakdown_percentage = [100 * float(np.sum(m.real_repair_time)) / manuf_line.sim_time for m in manuf_line.list_machines]
            if st.session_state.configuration["dev_mode"]:
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
            st.plotly_chart(fig, use_container_width=True)


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

        # Plot robots waiting times
        with c5:
            fig_m, ax_m = plt.subplots()
            ax_m.set_ylabel('Percentage (%)')
            
            # Calculate machine efficiency rate, machine available percentage, breakdown percentage, and waiting time percentage
            waiting_times_robots = [100*r.waiting_time / manuf_line.sim_time for r in manuf_line.robots_list]
            operating_times_robots = [100 - waiting for waiting in waiting_times_robots]

            chart_data = {
                "Robot": [robot.ID for robot in manuf_line.robots_list],
                "Operating": operating_times_robots,
                "Waiting": waiting_times_robots,
            }

            # Convert to DataFrame
            fig = go.Figure()

            # Add bar traces for each utilization type
            fig.add_trace(go.Bar(x=chart_data["Robot"], y=chart_data["Operating"], name="Operating", marker_color="green"))
            fig.add_trace(go.Bar(x=chart_data["Robot"], y=chart_data["Waiting"], name="Waiting", marker_color="orange"))

            # Update layout
            fig.update_layout(
                title="Robot Utilization Rate",
                xaxis_title="Robot",
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


        col_operator_work, col_central_storage = st.columns(2)
        
        # Operator Plot
        with col_operator_work:
            fig = go.Figure()

            # Add bar traces for each utilization type
            op_WCs = []
            for op in manuf_line.manual_operators:
                op_WCs.append(op.wc)
            fig.add_trace(go.Bar(x=[op.id for op in manuf_line.manual_operators], y=op_WCs, name="Cumulated WC", marker_color="green"))

            # Update layout
            fig.update_layout(
                title="Operator Work Content",
                xaxis_title="Operator",
                yaxis_title="Cumulated Work Content (s)",
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
        
        # Central storage plot
        if st.session_state.configuration["central_storage_enable"]:
            with col_central_storage:

                all_references = manuf_line.central_storage.all_allowed_references

                # Data required : A name for each block, number of items of each reference for every block, remaining space for every block.
                chart_data = {ref: [] for ref in all_references}
                chart_data["Blocks"] = []
                chart_data["Remaining space"] = []

                # Retrieving data from the central storage
                for side in manuf_line.central_storage.stores:
                    for i, block in enumerate(manuf_line.central_storage.stores[side]):
                        
                        # Add the name of the block
                        chart_data["Blocks"].append(f"{side.capitalize()}_block_{i+1}") 

                        # Count the number of occurrences in the block of each reference, even the ones that are not allowed (value will then be 0)
                        for ref in all_references:
                            chart_data[ref].append([ref_data['name'] for ref_data in block['store'].items].count(ref))

                        # Count the remaining space
                        chart_data["Remaining space"].append(block['store'].capacity - len(block['store'].items))


                fig = go.Figure()

                # Add bar traces for each reference and the remaining space
                for i, ref in enumerate(all_references):
                    fig.add_trace(go.Bar(x=chart_data["Blocks"], y=chart_data[ref], name=ref.capitalize()))
                
                fig.add_trace(go.Bar(x=chart_data["Blocks"], y=chart_data["Remaining space"], name="Remaining", marker_color="rgba(255, 165, 0, 0.5)"))

                # Update layout
                fig.update_layout(
                    title="Central storage",
                    xaxis_title="Blocks",
                    yaxis_title="Number of items",
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
       
    def assembly_section(self):

        self.placeholder = st.empty()

        # if(st.session_state['staticoptim_pressed'] == 1):
        #     with self.placeholder.container():
        uploaded_file_mbom = st.file_uploader("Upload Workplan", type=["xml"])
        tab1, tab2 = st.tabs(["Assembly Tasks", "Parts List"])
        with tab1:
            columns_to_keep = ['id', 'cycleTime',  'type', 'assy', 'precedency', 'forbidden','weight'] 
            #uploaded_file_line_data = st.file_uploader("Upload Production Line Data", type=["xlsx", "xls"])
            st.subheader("Assembly Tasks")
            if hasattr(st.session_state, 'mbom_data') and  isinstance(st.session_state.mbom_data, pd.DataFrame):
                st.session_state.mbom_data = st.session_state.mbom_data[columns_to_keep]

                updated_df = st.data_editor(st.session_state.mbom_data, num_rows="dynamic", key="tasks_edit")
                if st.button("Save Data", key="static_data_save"):
                    st.session_state.mbom_data = updated_df.copy()
                    st.rerun()

        with tab2:
            #uploaded_file_line_data = st.file_uploader("Upload Production Line Data", type=["xlsx", "xls"])
            st.subheader("Parts to be assembled")
            if hasattr(st.session_state, 'parts_data') and  isinstance(st.session_state.parts_data, pd.DataFrame):
                updated_df = st.data_editor(st.session_state.parts_data, num_rows="dynamic", key="parts_edit")
                if not st.session_state.parts_data.equals(updated_df):
                    st.session_state.parts_data = updated_df.copy()
                    st.rerun()
        
        with st.expander("Customize the production cost - CapEx & OpEx"):
            columns = st.columns(2)
            with columns[0]:
                st.session_state.configuration_static["Machine Cost"] = st.text_input("Machine Cost", value=st.session_state.configuration_static.get("Target CT", "100"))
                st.session_state.configuration_static["Type of Machine"] = st.selectbox("Type of Machine", ["V-Cell", ""], index=0 if st.session_state.configuration_static.get("Type of Machine") == "V-cell" else 1)
            with columns[1]:
                st.session_state.configuration_static["Operator Cost"] = st.text_input("Operator Cost", value=st.session_state.configuration_static.get("Target CT", "100"))
                st.session_state.configuration["MISC Cost"] = st.text_input("MISC Costs", value=st.session_state.configuration_static.get("Tolerance", "0.1"))

        with st.expander("Customize the optimization model", expanded=True):
            columns = st.columns(2)
            with columns[0]:
                st.session_state.configuration_static["Target CT"] = st.text_input("Target MCTO", value=st.session_state.configuration_static.get("Target CT", "100"))
                st.session_state.configuration_static["Search Speed"] = st.selectbox("Search Speed", ["Moderate", "Fast", "Slow"], index = ["Moderate", "Fast", "Slow"].index(st.session_state.configuration_static.get("Search Speed", "Fast")) )
                #st.session_state.configuration_static["reset_shift"] = st.checkbox("Enable Shift Reseting", value=st.session_state.configuration_static.get("reset_shift", False))
            with columns[1]:
                st.session_state.configuration["Tolerance"] = st.text_input("Tolerance", value=st.session_state.configuration_static.get("Tolerance", "0.1"))
                st.session_state.configuration_static["Exploration Mode"] = st.selectbox("Exploration Mode", ["Standard", "Out-Of-the-Box"], index=0 if st.session_state.configuration_static.get("Exploration Mode") == "Standard" else 1)


        if st.button("Run Sequence Generation"):
            progress_text = "Operation in progress. Please wait."
            self.my_bar_static_optim = st.progress(0, text=progress_text)

            if isinstance(uploaded_file_mbom, str):
                Tasks = read_prepare_mbom(uploaded_file_mbom, uploaded_file_mbom)
            else:
                Tasks = read_prepare_mbom(st.session_state.mbom_data, st.session_state.parts_data)
            
        
            

            if st.session_state.configuration_static["Exploration Mode"] == "Standard":
                tolerance = 0.1
            if st.session_state.configuration_static["Exploration Mode"] == "Out-Of-the-Box":
                tolerance = 0.5
            else:
                tolerance = 0.1
            
            target_CT = float(st.session_state.configuration_static["Target CT"])

            if st.session_state.configuration_static["Search Speed"] == "Fast":
                N_episodes = 10000
                xml_file = 'assets\inputs\L76 Dual Passive MBOM.xml'
                max_cycle_time = target_CT/2 +5 
                best_solution, ressource_list, operators_list, session_rewards = schedule_tasks(xml_file, max_cycle_time, tolerance)

            elif st.session_state.configuration_static["Search Speed"] == "Slow":
                N_episodes = 1000000
                best_solution, ressource_list, operators_list, session_rewards = run_QL(N_episodes, Tasks, target_CT, tolerance, self)

            else:
                N_episodes = 10000
                best_solution, ressource_list, operators_list, session_rewards = run_QL(N_episodes, Tasks, target_CT, tolerance, self)

            
            

            # Print the task allocation to machines
            # for i, machine in enumerate(machines):
            #     print(f"Machine {i+1}: Tasks: {machine}, Total Cycle Time: {machine_loads[i]}")

            if st.session_state.configuration["dev_mode"]:
                print("Best Soluton = ", best_solution)
                print("Machines = ", ressource_list[1])
                print("Operators = ", operators_list[1])
            
            print("Best Soluton = ", best_solution)
            print("Machines = ", ressource_list)
            print("Operators = ", operators_list)
            self.my_bar_static_optim.empty()
            st.header("Results")
            col = st.columns(4, gap='medium')
            with col[0]:
                CTs_pertwo = [ressource_list[1][i] + ressource_list[1][i + 1] for i in range(0, len(ressource_list[1]) - 1, 2)] 
                # Append the last element if the list has an odd length 
                if len(ressource_list[1]) % 2 != 0: 
                    CTs_pertwo.append(ressource_list[1][-1]) 
                global_cycle_time= max(CTs_pertwo)
                delta_target = (target_CT-float(global_cycle_time))/float(target_CT)
                st.metric(label="# Estimated Machine CT", value=str(int(global_cycle_time))  +" s", delta=f"{delta_target:.2%}")
            
            with col[1]:
                n_machines =ressource_list[0]//2
                st.metric(label="# N. of Machines", value=str(int(n_machines)))

            with col[2]:
                global_cycle_time= target_CT
                st.metric(label="# N. of Operators", value=str(int(operators_list[0])))
        
            
            st.markdown("""---""")
            # c11, c12, c13= st.columns([0.5,0.3,0.2])
            # c3, c4= st.columns([0.5,0.5])
            #with c11:
            
            st.markdown("### Best Sequence Overview")
            ### TODO: Should avoid repeating the components, if already done, then call it SubAssy OP01
            workstations = parts_to_workstation(st.session_state.mbom_data, st.session_state.parts_data, best_solution)

            #stations, workstations = parts_to_workstation_n(st.session_state.mbom_data, st.session_state.parts_data, best_solution)

            #Display each workstation and its corresponding parts
            for workstation, parts in sorted(workstations.items()):
                st.markdown(f"###### OP 0{workstation}")

                # Retrieve the thumbnail for each part by matching the part reference to parts_data
                part_thumbnails = []
                for part_ref in parts:
                    # Get the thumb URL from parts_data for this part_ref
                    part_info = st.session_state.parts_data[st.session_state.parts_data['ref'] == part_ref]

                    if not part_info.empty: # An isolated part
                        thumb_url = str(part_info['thumb'].values[0])  # Get the first (and only) matching thumb
                        part_name = str(part_info['PartFamily'].values[0]) +" "+ part_ref

                        # Add the thumb to the list (thumb_url and part_ref as caption)
                        part_thumbnails.append((thumb_url, part_name))
                    else: # An OP subassy
                        thumb_url = str(part_ref) 
                        part_name = part_ref
                        part_thumbnails.append((thumb_url, part_name))


                if part_thumbnails:
                    images, captions = zip(*part_thumbnails) 
                    try:
                        st.image(list(images), list(captions), width=100)  
                    except:
                        unknown_images= [".//assets//icons//unknown-part.png" for _ in range(len(images))]
                        st.image(list(unknown_images), list(captions), width=100) 


            st.markdown("### Detailed Results")
            fig = go.Figure()

            CT_machines = [ressource_list[1][i]+ressource_list[1][i-1] for i in range(len(ressource_list[1])) if i%2!=0]
            fig.add_trace(go.Bar(x=["M"+str(i+1) for i in range(len(CT_machines))], y=list(CT_machines), name='Global CT', marker_color='green'))

            # Update layout
            fig.update_layout(
                title='Machine Times per Machine',
                xaxis_title='Machines ID',
                yaxis_title='Machine Time | MT (s)',
                margin=dict(l=0, r=0, t=30, b=20)
            )

            st.plotly_chart(fig, use_container_width=True)


            fig = go.Figure()

                #fig.add_trace(go.Scatter(x=list(range(len(machines_CT[-1]))), y=machines_CT[-1], mode='lines', name=machine.ID))
                #fig.add_trace(go.Scatter(x=[t[0] for t in manuf_line.machines_output[i]], y=[t[1] for t in manuf_line.machines_output[i]], mode='lines', name=machine.ID))
            fig.add_trace(go.Scatter(x=list(range(len(session_rewards))), y=session_rewards, mode='lines', name='Global CT', marker_color='green'))

            fig.update_layout(
                title='Evolution of Sequence Scores',
                xaxis_title='Iterations',
                yaxis_title='Desirability Score',
                margin=dict(l=0, r=0, t=30, b=20)
            )

            st.plotly_chart(fig, use_container_width=True)
            time.sleep(1)
    
            # Generate stations
            station_ids = generate_station_ids(len(CTs_pertwo)) + ["EOL1"]
            print("station N = ", station_ids)
            # Create empty DataFrame for table
            table_data = pd.DataFrame({
                "Station ID": station_ids,
                #"Automatic": [0] * len(station_ids),  ressource_list[1]
                "Automatic": ressource_list[1][:len(station_ids)-1] + [0],
                "Operator ID": [None] * len(station_ids),  # Empty for operators, to be selected
                "Manual Time": [10] * len(station_ids)  # Default manual time as 0
            })

            # Assign operators based on input
            operators = [f"OP{i}" for i in range(1, len(operators_list) + 1)]
            table_data["Operator ID"] = [operators[i % len(operators_list)] for i in range(len(station_ids))]

            st.write("Add End of Line Stations if Needed.")
            edited_table = st.data_editor(table_data, num_rows="dynamic")

            
            manu_op_assignments = {}

            # Iterate over the edited table to populate manu_op_assignments
            for index, row in edited_table.iterrows():
                station_id = row["Station ID"]
                operator_id = int(row["Operator ID"].replace('OP', ''))  # Extract the operator number from "OPx"
                manual_time = row["Manual Time"]
                
                # Assign the station ID to the (operator_id, manual_time) tuple
                manu_op_assignments[station_id] = (operator_id, manual_time)

            st.session_state.static_optim_config = {
                "takt_time": str(target_CT),
                "n_machines": n_machines,
                "n_stations_permachine": 2,
                "list_operators": operators_list,
                "ressource_list": ressource_list[1],
                "manu_op_assignments": manu_op_assignments
            }
            st.session_state.static_best_solution = best_solution
            st.button("Run", on_click=self.click_detail_static_optim)

        if st.session_state.detail_btn_run:

            manu_op_assignments = st.session_state.static_optim_config['manu_op_assignments']    

            # self.save_global_settings(self.manuf_line)
            st.session_state.configuration = {
            "sim_time": "3600*24*1",
            "takt_time": st.session_state.static_optim_config['takt_time'],
            "enable_robots": False,
            "strategy": "Balanced Strategy",
            "reset_shift": False,
            "dev_mode":False,
            "stock_capacity": "10000000",
            "safety_stock": "1",
            "n_repairmen": 3,
            "enable_random_seed": True,
            "enable_breakdowns": False,
            "breakdown_dist_distribution": "Weibull Distribution",
            "central_storage_enable": False,
            "central_storage_ttr": {'front': 100, "back": 100},
            }
            

            st.session_state.line_data, st.session_state.multi_ref_data = prepare_detailed_line_sim(st.session_state.static_optim_config['ressource_list'], [25], manu_op_assignments)

            env = simpy.Environment()
            tasks = []
            config_file = 'config.yaml'
            self.manuf_line = ManufLine(env, tasks, config_file=config_file)
            self.manuf_line.references_config = st.session_state.multi_ref_data.set_index('Machine').to_dict(orient='list')
            self.manuf_line.machine_config_data = st.session_state.line_data.values.tolist()
            self.manuf_line.save_global_settings(st.session_state.configuration, self.manuf_line.references_config, self.manuf_line.machine_config_data, buffer_sizes=[])

            self.manuf_line.create_machines(st.session_state.line_data.values.tolist())
            self.all_prepared = True
            st.session_state.configuration["central_storage_enable"] = False
            st.session_state.configuration['enable_robots'] = False

            self.manuf_line.dev_mode = False
            self.run_simulation(self.manuf_line)
            st.markdown("### Recap - Sequence Overview")
            workstations = parts_to_workstation(st.session_state.mbom_data, st.session_state.parts_data, st.session_state.static_best_solution)

            #stations, workstations = parts_to_workstation_n(st.session_state.mbom_data, st.session_state.parts_data, best_solution)

            #Display each workstation and its corresponding parts
            for workstation, parts in sorted(workstations.items()):
                st.markdown(f"###### OP 0{workstation}")

                # Retrieve the thumbnail for each part by matching the part reference to parts_data
                part_thumbnails = []
                for part_ref in parts:
                    # Get the thumb URL from parts_data for this part_ref
                    part_info = st.session_state.parts_data[st.session_state.parts_data['ref'] == part_ref]

                    if not part_info.empty: # An isolated part
                        thumb_url = str(part_info['thumb'].values[0])  # Get the first (and only) matching thumb
                        part_name = str(part_info['PartFamily'].values[0]) +" "+ part_ref

                        # Add the thumb to the list (thumb_url and part_ref as caption)
                        part_thumbnails.append((thumb_url, part_name))
                    else: # An OP subassy
                        thumb_url = str(part_ref) 
                        part_name = part_ref
                        part_thumbnails.append((thumb_url, part_name))


                if part_thumbnails:
                    images, captions = zip(*part_thumbnails) 
                    try:
                        st.image(list(images), list(captions), width=100)  
                    except:
                        unknown_images= [".//assets//icons//unknown-part.png" for _ in range(len(images))]
                        st.image(list(unknown_images), list(captions), width=100) 

            st.session_state.detail_btn_run = False
            #st.session_state['staticoptim_pressed'] = 1

                
        return True

    def run_main(self):
        #st.set_page_config(page_title="PRODynamics", page_icon="⚙️", layout="wide")

        with st.sidebar:
            selected = option_menu(
            menu_title = "Prodynamics",
            options = ["Home","Global Settings","Process Data","Simulation Lab","Buffer Sizing Lab","Static Assembly Lab","Contact Us"],
            icons = ["house","gear","activity","play-circle-fill","bar-chart-line-fill", "bounding-box","question-circle-fill"],
            menu_icon = "cast",
            default_index = 0,
        )
            
        if selected == "Home":
            self.home()
        elif selected == "Global Settings":
            self.global_configuration()
        elif selected == "Process Data":
            self.process_data()
        elif selected == "Simulation Lab":
            self.simulation_page()
        elif selected == "Buffer Sizing Lab":
            self.optimization_page()
            pass
        elif selected == "Static Assembly Lab":
            self.assembly_section()
            pass
        elif selected == "Contact Us":
            # Rebuild a new page with different contacts (For dev, business & process)
            pass
  

if __name__ == "__main__":
    app = PRODynamicsApp()
    app.run_main()
