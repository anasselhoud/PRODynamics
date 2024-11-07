import random
import simpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from matplotlib.animation import FuncAnimation
import tkinter as tk
import threading
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
from collections import deque
#from chart_studio.widgets import GraphWidget


env = simpy.Environment()


def generate_random_tasks(num_tasks):
    tasks = []
    for task_id in range(1, num_tasks + 1):
        machine_time = random.randint(1, 50)  # Adjust the range as needed
        #manual_time = random.randint(1, 50)  # Adjust the range as needed
        manual_time = 0
        task = Task(task_id, machine_time, manual_time)
        tasks.append(task)
    return tasks

def known_tasks(list):
    tasks = []
    for task_id, task_mt in enumerate(list):
        machine_time = task_mt  # Adjust the range as needed
        manual_time = 3 # Adjust the range as needed
        task = Task(task_id, machine_time, manual_time)
        tasks.append(task)
    return tasks

def update(frame, buffer_states):
    plt.clf()  # Clear the previous frame
    plt.bar(range(len(buffer_states)), buffer_states)
    plt.title(f"Time Step: {frame}")
    plt.xlabel("Buffer Index")
    plt.ylabel("Buffer State")

#tasks = generate_random_tasks(5)
#tasks = known_tasks([31, 18, 42, 36, 20])
def run(assembly_line):
    assembly_line.run()
    assembly_line.get_results()
    list_machines = assembly_line.get_track()


def clock(env, assembly_line):

    shift_ct = []
    global_ct = []
    sim_time = []
    constant_x_values = []
    d_sim_time = deque(maxlen=250)
    d_shift_ct = deque(maxlen=250)
    d_global_ct = deque(maxlen=250)
    d_constant_x_values = deque(maxlen=250)
    global last_breakdown 
    last_breakdown = 0
    text = canvas.create_text(10, 10, fill='black', anchor='nw', font=('Arial', 12), text='Time = %d seconds' % env.now)
    text2 = canvas.create_text(10, 30, fill='black', anchor='nw', font=('Arial', 12), text='Parts = %d' % assembly_line.list_machines[-1].buffer_out.level)
    text4 = canvas.create_text(10, 50, fill='black', anchor='nw', font=('Arial', 12), text='Shift Cycle Time = %.2f seconds' % 0)
    text3 = canvas.create_text(10, 70, fill='black', anchor='nw', font=('Arial', 12), text='Global Cycle Time = %.2f seconds' % 0)

    canvas2 = tk.Canvas(window, width=320, height=80, bg="white")
    canvas2.pack(side=tk.TOP, anchor=tk.NW) 


    speedup_btn = tk.Button(canvas2, text="+ Speed Up", command=increase_timeout, bg='#4CAF50', fg='white', padx=10, pady=5, borderwidth=2, relief=tk.GROOVE)
    speedup_btn.pack(side=tk.LEFT, padx=5)  # Use side=tk.LEFT to pack the button to the left of the previous one

    slowdown_btn = tk.Button(canvas2, text="- Slow Down", command=decrease_timeout, bg='#F44336', fg='white', padx=10, pady=5, borderwidth=2, relief=tk.GROOVE)
    slowdown_btn.pack(side=tk.LEFT, padx=5)  # Use side=tk.LEFT to pack the button to the left of the previous one
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), sharex=True)
    canvas3 = tk.Canvas(window, width=820, height=720, bg='white')
# Update the subplot titles
    #fig.update_yaxes(title_text='Shift Cycle Time (s)', row=1, col=1)
    #fig.update_yaxes(title_text='Global Cycle Time (s)', row=2, col=1)


    canvas_fig = FigureCanvasTkAgg(fig, master=canvas3)
    canvas_fig_widget = canvas_fig.get_tk_widget()
    canvas_fig_widget.pack(side=tk.LEFT, fill=tk.BOTH)
    toolbar = NavigationToolbar2Tk(canvas_fig,canvas3)
    toolbar.update()
    canvas3.pack(expand=tk.YES, fill=tk.BOTH)
    buffer_canvas = tk.Canvas(canvas3, width=400, height=120*len(assembly_line.list_machines))  # Adjust the width as needed
    buffer_canvas.pack(side=tk.RIGHT, fill=tk.BOTH)
    global timeout
    timeout = 1
    def update_display():
        elapsed_seconds = env.now
        
        elapsed_seconds_shift = elapsed_seconds % (3600 * 8)
        elapsed_time_str = format_time(elapsed_seconds)
        canvas.itemconfigure(text, text='Time = %s' % elapsed_time_str)
        canvas.itemconfigure(text2, text='Parts = %d' % assembly_line.list_machines[-1].parts_done)
        new_breakdown = None
        global last_breakdown
        
        for m in assembly_line.list_machines:
            
            if m.broken and env.now-last_breakdown>float(assembly_line.breakdowns["mttr"]):
                print("Machine " + m.ID + " broken")
                new_breakdown = env.now
                last_breakdown = new_breakdown
                break
        if elapsed_seconds > 0 and assembly_line.list_machines[-1].buffer_out.level > 0 and all([machine.parts_done_shift > 0 for machine in assembly_line.list_machines]) :
            if elapsed_seconds_shift == 0:
                elapsed_seconds_shift = 100
            shift_cycle_time = np.max([elapsed_seconds_shift / (assembly_line.list_machines[-1].parts_done_shift) for i in range(len(assembly_line.list_machines))])
                
            canvas.itemconfigure(text4, text='Shift Cycle Time = %.2f seconds' % shift_cycle_time)
            cycle_time = elapsed_seconds / assembly_line.list_machines[-1].parts_done
            canvas.itemconfigure(text3, text='Global Cycle Time = %.2f seconds' % cycle_time)

            if elapsed_seconds > 500: #avoid warm-up
                update_buffer_viz(buffer_canvas, assembly_line)
                if new_breakdown != None:
                    update_plot_CT((shift_cycle_time, env.now), (cycle_time, env.now), new_breakdown)
                    new_breakdown = None
                else:
                    update_plot_CT((shift_cycle_time, env.now), (cycle_time, env.now))
                
            
    def update_plot_CT(new_y, new_y2, new_breakdown=None):
        
        
        warm_up_period = 1000
        warm_up_passsed = False
        window_size = 3600*24*2
        
        shift_ct.append(new_y[0])
        global_ct.append(new_y2[0])
        sim_time.append(new_y[1])
        d_sim_time.append(new_y[1])
        d_shift_ct.append(new_y[0])
        d_global_ct.append(new_y2[0])
        index_warm_up = find_closest_index(sim_time, warm_up_period)
        axs[0].clear()
        axs[1].clear()
        if new_breakdown != None: 
            constant_x_values.append(new_y[1])
            d_constant_x_values.append(new_y[1])

        # Plot the data
        axs[0].plot(list(d_sim_time), list(d_shift_ct), label='Shift Cycle Time')
        axs[1].plot(list(d_sim_time), list(d_global_ct), color='orange', label='Global Cycle Time')
 
        axs[0].set_ylabel('Shift Cycle Time (s)')
        axs[0].set_title('Shift Cycle Time (s)')
        axs[1].set_ylabel('Global Cycle Time (s)')
        axs[1].set_title('Avg. Annual Cycle Time(s)')
        fig.text(0.5, 0.04, 'Duration (s)', ha='center', va='center')
        for x_value in list(d_constant_x_values):
            axs[0].axvline(x=x_value, color='red', linestyle='--')
            axs[1].axvline(x=x_value, color='red', linestyle='--')

        if max(sim_time) > warm_up_period and not warm_up_passsed:
            for ax in axs:
                ax.set_xlim(warm_up_period, max(list(d_sim_time)))

        if  len(d_sim_time) == d_sim_time.maxlen:
            for ax in axs:
                ax.set_xlim(min(list(d_sim_time)), max(list(d_sim_time)))
        
        #     axs[0].set_ylim(min(shift_ct[index_warm_up:])-50, max(shift_ct[index_warm_up:])+50)
        #     axs[1].set_ylim(min(global_ct[index_warm_up:])-50, max(global_ct[index_warm_up:])+50)
        #     warm_up_passsed = True
    
            

        # for ax in axs:
        #     ax.relim()
        #     ax.autoscale_view()
        # Draw the updated plot on the canvas
        canvas_fig.draw()

    def clock_generator():
        while True:
            global timeout
            yield env.timeout(timeout)
            update_display()

    return clock_generator()

def increase_timeout():
    global timeout
    timeout = timeout + 10

def decrease_timeout():
    global timeout
    if timeout > 10:
        timeout = timeout - 10
    if timeout < 10:
        timeout = 1



if __name__ == '__main__':

    #animation = FuncAnimation(plt.gcf(), update, frames=len(buffer), fargs=(buffer,), interval=100, repeat=False)
    window = tk.Tk()
    window.title("ManufactSim Dynamics")
    window.geometry("1280x720")
    canvas = tk.Canvas(window, width=320, height=100, bg="#3D59AB")
    canvas.pack(side=tk.TOP, anchor=tk.NW) 
    #ledButton = tk.Button(window, text = "   ",state=tk.DISABLED, height = 2, width =8,bg="red")
    #ledButton.pack()
    #plt.show()
    df_tasks = pd.read_xml('./workplan_TestIsostatique_modified.xml', xpath=".//weldings//welding")
    tasks = known_tasks(df_tasks["cycleTime"].astype(int).tolist())
    config_file = 'config.yaml'
    task_assignement = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3,  3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4]
    assembly_line = ManufLine(env, 4, tasks, [[1], [2], [3, 4]], task_assignement, config_file=config_file)
    assembly_line.set_CT_machines([20,0,0,0])
    env.process(clock(env, assembly_line))
    thread = threading.Thread(target=run, args={assembly_line,})
    thread.start()
    
    
    window.mainloop()

