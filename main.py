"""
Each Part has consecutive set of Processes to be executed
Each Process requires Resource

Simulation Components follow the naming convention below.

[6Factor Concept]   [Class Object]
Part                Job
Process             Operation
Resource            Machine, Worker, Factory, Line, Transporter, etc.
Source              Source
Sink                Sink
Monitor             Monitor

Based on 'JSSP_6Factors_nobuffer_231113.py' file
Revised in 2023. 11. 15.
"""
from environment.Source import Source
from environment.Sink import Sink
from environment.Part import Job, Operation
from environment.Process import Process
from environment.Resource import Machine
from environment.Monitor import Monitor
from postprocessing.PostProcessing import *
from data import *
from visualization.Gantt import *
from visualization.GUI import GUI

import simpy, os, random
import pandas as pd
import numpy as np
from collections import OrderedDict



if __name__ == "__main__":

    env = simpy.Environment()
    monitor = Monitor(filepath)
    model = dict()
    for i in range(NUM_MACHINE):
        model['Source' + str(i)] = Source(env, 'Source' + str(i), model, monitor,
                                          part_type=i, IAT=IAT, num_parts=float('inf'))
        model['Process' + str(i)] = Process(env, 'Process' + str(i), model, monitor, solution_machine_order[i],
                                            capacity=1, in_buffer=12, out_buffer=12)
        model['M' + str(i)] = Machine(env, i)
    model['Sink'] = Sink(env, monitor)

    # In case of the situation where termination of the simulation greatly affects the machine utilization time,
    # it is necessary to terminate all the process at (SIMUL_TIME -1) and add up the process time to all machines
    env.run(SIMUL_TIME)
    monitor.save_event_tracer()
    machine_log = machine_log(filepath)
    gantt = Gantt(machine_log, len(machine_log), printmode=True, writemode=True)
    gui = GUI(gantt)
    print()




