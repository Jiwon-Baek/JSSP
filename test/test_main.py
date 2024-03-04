from environment.Source import Source
from environment.Sink import Sink
from environment.Process import Process
from environment.Resource import Machine
from environment.Monitor import Monitor
from postprocessing.PostProcessing import *
from data import *
from visualization.Gantt import *
from visualization.GUI import GUI
import simpy, os

"""
Test Case에 대해 monitor class log, Gantt Chart, GUI 창을 띄우는 코드
data.py에 있는 21번째 줄의 경로를 './abz5.csv' 로 바꿔주세요. (20번째 줄 주석 해제)
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
        model['Process' + str(i)] = Process(env, 'Process' + str(i), model, monitor, solution_machine_order[i],
                                            capacity=1, in_buffer=12, out_buffer=12)
        model['M' + str(i)] = Machine(env, i)
    # print('471')
    model['Sink'] = Sink(env, monitor)

    # In case of the situation where termination of the simulation greatly affects the machine utilization time,
    # it is necessary to terminate all the process at (SIMUL_TIME -1) and add up the process time to all machines

    # 어떤 machine M1이 t=1000부터 t=2500까지 작업 중이라면 임의로 simul_time = 2000까지 지정해서 실행할 경우,
    # 이 machine의 작업 log는 t=1000에 시작한 작업이 반영되지 않아서 utilization이 50% 미만으로 계산될 수 있음.
    # 따라서 작업이 완료되지 않았더라도 t=1000~1999까지는 작업중이었다는 것을 명시하는 코드가 추가로 필요함. (아직 반영 전)

    env.run(SIMUL_TIME)
    monitor.save_event_tracer()
    machine_log = machine_log(filepath)
    gantt = Gantt(machine_log, len(machine_log), printmode=True, writemode=True)
    gui = GUI(gantt)
    print()
