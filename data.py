import pandas as pd
import os

"""
abz5 Problem
"""

current_working_directory = os.getcwd()
op_data = []

"""
Please make sure the filepath is correct!
"""
def set_problem(problem, problem_path):

    # Directory Configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
    filepath = os.path.join(script_dir,problem_path)

    data = pd.read_csv(filepath + problem+'.txt', skiprows=[0], header=None, encoding='cp949', delimiter='\t')

    for i in range(10):
        op_data.append([])
        for j in range(10):
            op_data[i].append((data.iloc[10+i,j]-1, data.iloc[i, j]))




# solution = pd.read_csv(filepath + '/abz5_solution_start.csv', header=None)
# solution_machine_order = [[] for i in range(10)]
# for i in range(10):
#     start_time = list(solution.iloc[:, i])
#     value = sorted(start_time, reverse=False)
#     solution_machine_order[i] = sorted(range(len(start_time)), key=lambda k: start_time[k])
