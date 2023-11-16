import os
from datetime import datetime
from data import op_data

# Time Variables
# IAT = 'exponential(25)'
IAT = 30
SIMUL_TIME = 120

# Monitor Variables
OBJECT = 'Single Part'
CONSOLE_MODE = True

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
filepath = os.path.join(save_path, filename+'.csv')



# # Job 개수와 Machine 개수가 일치하지 않을 때 Max값을 찾는 코드 (Job이 10개인데 Machine은 15대를 쓴다던가, 등)
NUM_MACHINE = 0
for i in range(len(op_data)):
    for j in range(len(op_data[i])):
        # print(jobs_data[i][j][0])
        max_machine = op_data[i][j][0]
        if max_machine > NUM_MACHINE:
            NUM_MACHINE = max_machine
NUM_MACHINE += 1

n_op = 25
n_show = n_op + 1
show_interval_time = 100
finished_pause_time = 1000