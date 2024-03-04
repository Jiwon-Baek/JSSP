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


def calculate_score(x_array, y_array):
    score = [0.0 for i in range(4)]
    for i in range(len(x_array)):
        score[0] += kendall_tau_distance(x_array[i], y_array[i])
        score[1] += spearman_rank_correlation(x_array[i], y_array[i])
        score[2] += spearman_footrule_distance(x_array[i], y_array[i])
        score[3] += MSE(x_array[i], y_array[i])
        # score[3] += bubble_sort_distance(x_array[i])
    return score


class Individual():
    def __init__(self, seq):
        self.seq = seq
        self.job_seq = self.get_repeatable()
        self.feasible_seq = self.get_feasible()
        self.machine_seq = self.get_machine_order()
        self.MIO = []
        self.MIO_sorted = []
        self.makespan, self.mio_score = self.evaluate(self.machine_seq)
        self.score = calculate_score(self.MIO, self.MIO_sorted)

    def get_repeatable(self):
        cumul = 0
        sequence_ = np.array(self.seq)
        for i in range(NUM_MACHINE):
            for j in range(NUM_OP):
                sequence_ = np.where((sequence_ >= cumul) &
                                     (sequence_ < cumul + NUM_MACHINE), i, sequence_)
            cumul += NUM_MACHINE
        sequence_ = sequence_.tolist()
        return sequence_

    def get_feasible(self):
        temp = 0
        cumul = 0
        sequence_ = np.array(self.seq)
        for i in range(NUM_MACHINE):
            idx = np.where((sequence_ >= cumul) & (sequence_ < cumul + NUM_MACHINE))[0]
            for j in range(NUM_OP):
                sequence_[idx[j]] = temp
                temp += 1
            cumul += NUM_MACHINE
        return sequence_

    def get_machine_order(self):
        m_list = []
        for num in self.feasible_seq:
            idx_i = num % NUM_OP
            idx_j = num // NUM_MACHINE
            m_list.append(op_data[idx_j][idx_i][0])
        m_list = np.array(m_list)

        m_order = []
        for num in range(NUM_MACHINE):
            idx = np.where((m_list == num))[0]
            job_order = [self.job_seq[o] for o in idx]
            m_order.append(job_order)
        return m_order

    def evaluate(self, machine_order):
        env = simpy.Environment()
        monitor = Monitor(filepath)
        model = dict()
        for i in range(NUM_MACHINE):
            model['Source' + str(i)] = Source(env, 'Source' + str(i), model, monitor,
                                              part_type=i, IAT=IAT, num_parts=float('inf'))
            model['Process' + str(i)] = Process(env, 'Process' + str(i), model, monitor, machine_order[i],
                                                capacity=1, in_buffer=12, out_buffer=12)
            model['M' + str(i)] = Machine(env, i)
        model['Sink'] = Sink(env, monitor)

        # In case of the situation where termination of the simulation greatly affects the machine utilization time,
        # it is necessary to terminate all the process at (SIMUL_TIME -1) and add up the process time to all machines
        env.run(SIMUL_TIME)
        # monitor.save_event_tracer()
        # machine_log_ = machine_log(filepath)
        # gantt = Gantt(machine_log_, len(machine_log_), printmode=True, writemode=True)
        # gui = GUI(gantt)
        for i in range(NUM_MACHINE):
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

    ind = Individual(optimal)
    makespan[NUM_ITERATION] = ind.makespan
    score[0][NUM_ITERATION] = ind.score[0]
    score[1][NUM_ITERATION] = ind.score[1]
    score[2][NUM_ITERATION] = ind.score[2]
    score[3][NUM_ITERATION] = ind.score[3]

    popul = []
    for i in range(NUM_ITERATION):
        seq = np.random.permutation(100)
        individual = Individual(seq)
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
