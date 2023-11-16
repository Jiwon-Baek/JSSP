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
from .environment import Source, Sink, Job, Operation, Process, Machine, Monitor
from config import *

import simpy, os, random
import pandas as pd
import numpy as np
from datetime import datetime
from collections import OrderedDict

save_path = '../result'
if not os.path.exists(save_path):
    os.makedirs(save_path)

if __name__ == "__main__":

    now = datetime.now()
    filename = now.strftime('%Y-%m-%d-%H-%M-%S')
    filepath = './result/'+filename+'.csv'
    env = simpy.Environment()
    monitor = Monitor(filepath)

    model = dict()

    for i in range(5):
        model['Source' + str(i)] = Source(env, 'Source' + str(i), model, monitor,
                                          part_type=i, IAT=IAT, num_parts=float('inf'))
        model['Process' + str(i)] = Process(env, 'Process' + str(i), model, monitor,
                                            capacity=1, in_buffer=12, out_buffer=12)
        model['M' + str(i)] = Machine(env, i)
    # print('471')
    model['Sink'] = Sink(env, monitor)

    # In case of the situation where termination of the simulation greatly affects the machine utilization time,
    # it is necessary to terminate all the process at (SIMUL_TIME -1) and add up the process time to all machines
    env.run(100)
    monitor.save_event_tracer()
