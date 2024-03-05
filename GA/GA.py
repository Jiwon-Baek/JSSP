import numpy as np

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

import simpy
from MachineInputOrder.utils import kendall_tau_distance, spearman_footrule_distance, spearman_rank_correlation, \
    bubble_sort_distance, MSE
def swap_digits(num):
    if num < 10:
        return num * 10
    else:
        units = num % 10
        tens = num // 10
        return units * 10 + tens


def calculate_score(x_array, y_array):
    score = [0.0 for i in range(6)]
    for i in range(len(x_array)):
        score[0] += kendall_tau_distance(x_array[i], y_array[i])
        score[1] += spearman_rank_correlation(x_array[i], y_array[i])
        score[2] += spearman_footrule_distance(x_array[i], y_array[i])
        score[3] += MSE(x_array[i], y_array[i])
        score[4] += bubble_sort_distance(x_array[i])
        correlation_matrix = np.corrcoef(x_array[i], y_array[i])
        score[5] += correlation_matrix[0,1]
        # score[3] += bubble_sort_distance(x_array[i])
    return score


class Individual():
    def __init__(self, num_machine=10, num_job=10, seq=None, solution_seq=None):
        if solution_seq != None:
            """
            5는 Job5의 0번째 operation
            15는 Job5의 1번째 operation
            이런 식으로 작성되어 있음
            사실상 [5 15 25 35 45 ... 95]까지가 한 Job을 나타냄
            """
            self.seq = self.interpret_solution(solution_seq)
        else:
            self.seq = seq  # 0부터 시작하는 값, op들의 순서

        self.num_machine = num_machine
        self.num_job = num_job
        self.job_seq = self.get_repeatable()
        self.feasible_seq = self.get_feasible()
        self.machine_seq = self.get_machine_order()
        self.MIO = []
        self.MIO_sorted = []
        self.makespan, self.mio_score = self.evaluate(self.machine_seq)
        self.score = calculate_score(self.MIO, self.MIO_sorted)
    def interpret_solution(self, s):
        # 리스트의 각 원소에 대해 숫자 바꾸기
        modified_list = [swap_digits(num) for num in s]
        return modified_list

    def get_repeatable(self):
        cumul = 0
        sequence_ = np.array(self.seq)
        for i in range(self.num_machine):
            for j in range(self.num_job):
                sequence_ = np.where((sequence_ >= cumul) &
                                     (sequence_ < cumul + self.num_machine), i, sequence_)
            cumul += self.num_machine
        sequence_ = sequence_.tolist()
        return sequence_

    def get_feasible(self):
        temp = 0
        cumul = 0
        sequence_ = np.array(self.seq)
        for i in range(self.num_machine):
            idx = np.where((sequence_ >= cumul) & (sequence_ < cumul + self.num_machine))[0]
            for j in range(self.num_job):
                sequence_[idx[j]] = temp
                temp += 1
            cumul += self.num_machine
        return sequence_

    def get_machine_order(self):
        m_list = []
        for num in self.feasible_seq:
            idx_i = num % self.num_job
            idx_j = num // self.num_machine
            m_list.append(op_data[idx_j][idx_i][0])
        m_list = np.array(m_list)

        m_order = []
        for num in range(self.num_machine):
            idx = np.where((m_list == num))[0]
            job_order = [self.job_seq[o] for o in idx]
            m_order.append(job_order)
        return m_order

    def evaluate(self, machine_order):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        folder_name = 'result'  # Define the folder name
        save_path = os.path.join(script_dir, folder_name)  # Construct the full path to the folder
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        now = datetime.now()
        filename = now.strftime('%Y-%m-%d-%H-%M-%S')
        filepath = os.path.join(save_path, filename + '.csv')
        env = simpy.Environment()
        monitor = Monitor(filepath)
        model = dict()
        for i in range(self.num_machine):
            model['Source' + str(i)] = Source(env, 'Source' + str(i), model, monitor,
                                              part_type=i, IAT=IAT, num_parts=float('inf'))
            model['Process' + str(i)] = Process(env, 'Process' + str(i), model, monitor, machine_order[i],
                                                capacity=1, in_buffer=12, out_buffer=12)
            model['M' + str(i)] = Machine(env, i)
        model['Sink'] = Sink(env, monitor)

        # In case of the situation where termination of the simulation greatly affects the machine utilization time,
        # it is necessary to terminate all the process at (SIMUL_TIME -1) and add up the process time to all machines
        env.run(SIMUL_TIME)

        monitor.save_event_tracer()
        # machine_log_ = machine_log(filepath)
        # gantt = Gantt(machine_log_, len(machine_log_), printmode=True, writemode=True)
        # gui = GUI(gantt)
        for i in range(self.num_machine):
            mio = model['M' + str(i)].op_where
            self.MIO.append(mio)
            self.MIO_sorted.append(np.sort(mio))

        mio_score = np.sum(np.abs(np.subtract(np.array(mio), np.array(sorted(mio)))))
        return model['Sink'].last_arrival, mio_score


if __name__ == "__main__":

    for i in range(10):
        op_data.append([])
        for j in range(10):
            op_data[i].append((data.iloc[10 + i, j] - 1, data.iloc[i, j]))

    print(os.path.dirname(os.path.abspath(__file__)))
    print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Get the directory of the current script

    filepath = os.path.join(project_dir, 'test/')
    data = pd.read_csv(filepath + 'abz5.csv', header=None)
    solution = pd.read_csv(filepath + '/abz5_solution_start.csv', header=None)

    optimal = solution.values.ravel()
    optimal = optimal.argsort(axis=0)

    NUM_ITERATION = 1000
    # machine_order = [[] for i in range(NUM_ITERATION)]
    makespan = [0 for i in range(NUM_ITERATION + 1)]
    score = [[0 for i in range(NUM_ITERATION + 1)] for i in range(4)]
    # for i in range(NUM_ITERATION):
    #     for j in range(10):
    #         temp = np.random.permutation(10)
    #         machine_order[i].append(temp.tolist())

    ind = Individual(10, 10, seq=optimal)
    makespan[NUM_ITERATION] = ind.makespan
    score[0][NUM_ITERATION] = ind.score[0]
    score[1][NUM_ITERATION] = ind.score[1]
    score[2][NUM_ITERATION] = ind.score[2]
    score[3][NUM_ITERATION] = ind.score[3]

    popul = []
    for i in range(NUM_ITERATION):
        seq = np.random.permutation(100)
        individual = Individual(10, 10, seq=seq)
        popul.append(individual)

        makespan[i] = individual.makespan
        score[0][i] = individual.score[0]
        score[1][i] = individual.score[1]
        score[2][i] = individual.score[2]
        score[3][i] = individual.score[3]

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    color = 'tab:red'
    ax1.set_ylabel('Kendall Tau')
    ax2.set_ylabel('spearman_rank')
    ax3.set_ylabel('spearman_footrule')
    ax4.set_ylabel('MSE')

    ax1.scatter(makespan, score[0], color=color, s=4)
    ax2.scatter(makespan, score[1], color=color, s=4)
    ax3.scatter(makespan, score[2], color=color, s=4)
    ax4.scatter(makespan, score[3], color=color, s=4)

    pearson_corr = [0.0 for i in range(4)]
    for i in range(4):
        correlation_matrix = np.corrcoef(makespan, score[i])
        pearson_corr[i] = correlation_matrix[0, 1]
    print('Pearson Correlation Coefficient')
    print('Kendall Tau :', pearson_corr[0])
    print('Spearman Rank :', pearson_corr[1])
    print('Spearman Footrule :', pearson_corr[2])
    print('MSE :', pearson_corr[3])
    # print('bubble_sort :',pearson_corr[3])
    plt.show()
