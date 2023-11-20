import pandas as pd
import os
# op_data = [  # ([list of available of machine], [corresponding process time])
#     [(0, 1), (1, 5), (2, 5), (3, 5), (4, 5)],  # part_type = 1
#     [(1, 2), (2, 5), (3, 5), (4, 5), (0, 5)],  # part_type = 2
#     [(2, 3), (3, 5), (4, 5), (0, 5), (1, 5)],  # part_type = 3
#     [(3, 4), (4, 5), (0, 5), (1, 5), (2, 5)],  # part_type = 4
#     [(4, 5), (0, 5), (1, 5), (2, 5), (3, 5)]  # part_type = 5
# ]


current_working_directory = os.getcwd()
op_data = []
# data = pd.read_csv('./test/abz5.csv', header=None)
data = pd.read_csv('./abz5.csv', header=None)

for i in range(10):

    op_data.append([])
    for j in range(10):
        op_data[i].append((data.iloc[10+i,j]-1, data.iloc[i, j]))


solution = pd.read_csv('./abz5_solution_start.csv', header=None)
solution_machine_order = [[] for i in range(10)]

for i in range(10):
    start_time = list(solution.iloc[:, i])
    value = sorted(start_time, reverse=False)
    solution_machine_order[i] = sorted(range(len(start_time)), key=lambda k: start_time[k])

print()
# TODO
"""
1. Machine별로 자신이 처리해야 하는 Part의 Sequence를 저장
2. 그 Sequence가 도착할 때까지 in_part queue에서 다음으로 넘어가지 않고 기다림
3. 해당 part가 오면 우선 처리하도록 코드 수정
"""

