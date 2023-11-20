from environment.Source import Source
from environment.Sink import Sink
from environment.Part import Job, Operation
from environment.Process import Process
from environment.Resource import Machine
from environment.Monitor import Monitor
from postprocessing.PostProcessing import *
from config import *
from visualization.Gantt import *
from visualization.GUI import GUI

import simpy, os, random
import pandas as pd
import numpy as np
from collections import OrderedDict


# TODO
"""
dispatch가 병렬로 돌아가지 않고 순차적으로만 돌아가서 일단 한 번 yield를 기다리는 중이라면 
새로 들어온 작업을 queue에 넣지 못하는 문제가 있는 것 같다
dispatch를 매번 새로운 part가 들어올 떄마다 한번 call 하는 하나의 함수로 만들어야겠다
"""
if __name__ == "__main__":

    # Directory Configuration
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the folder name
    folder_name = 'result'

    # Construct the full path to the folder
    save_path = os.path.join(script_dir, folder_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    now = datetime.now()
    filename = now.strftime('%Y-%m-%d-%H-%M-%S')
    filepath = os.path.join(save_path, filename + '.csv')

    env = simpy.Environment()
    monitor = Monitor(filepath)

    model = dict()

    for i in range(NUM_MACHINE):
        model['Source' + str(i)] = Source(env, 'Source' + str(i), model, monitor,
                                          part_type=i, IAT=IAT, num_parts=float('inf'))
        model['Process' + str(i)] = Process(env, 'Process' + str(i), model, monitor,
                                            capacity=1, in_buffer=12, out_buffer=12)
        model['M' + str(i)] = Machine(env, i)
    # print('471')
    model['Sink'] = Sink(env, monitor)

    # In case of the situation where termination of the simulation greatly affects the machine utilization time,
    # it is necessary to terminate all the process at (SIMUL_TIME -1) and add up the process time to all machines
    env.run(SIMUL_TIME)
    monitor.save_event_tracer()
    machine_log = machine_log(filepath)
    gantt = Gantt(machine_log, len(machine_log), printmode=True, writemode=True)
    gui = GUI(gantt)
    print()
