import simpy
from .Monitor import *
from config import *
from data import *

class Process(object):
    def __init__(self, _env, _name, _model, _monitor, _machine_order, capacity=float('inf'), priority=1, in_buffer=float('inf'),
                 out_buffer=float('inf')):
        # input data
        self.env = _env
        self.name = _name  # 해당 프로세스의 이름
        self.model = _model
        self.monitor = _monitor
        self.capa = capacity  # 해당 프로세스의 동시 작업 한도
        self.priority = priority  # 해당 프로세스의 우선 순위

        # if machine order is assigned (central scheduling)
        self.machine_order = _machine_order

        # variable defined in class
        self.parts_sent = 0
        self.scheduled = 0

        # buffer and machine
        self.in_part = simpy.FilterStore(_env, capacity=in_buffer + capacity)
        self.part_ready = simpy.FilterStore(_env, capacity=100)
        self.out_part = simpy.FilterStore(_env, capacity=out_buffer)

        # Part가 Process로 들어오는 것을 감지하기 위한 Event
        self.input_event = simpy.Event(_env)
        self.ready_event = simpy.Event(_env)
        self.route_ready = simpy.Event(_env)

        # get run functions in class
        # env.process(self.run())
        _env.process(self.work())
        _env.process(self.routing())
        _env.process(self.dispatch())

    def work(self):
        while True:
            # yield self.ready_event
            # self.ready_event = simpy.Event(self.env)
            # 1. check the compatible machine list
            part = yield self.part_ready.get()
            operation = part.op[part.step]
            yield operation.requirements  # Check if former operations are all finished

            # 2. check the machines(resources) that are suitable for the process.
            if isinstance(operation.machine_list, list):
                """
                Note: In the simple JSSP problem, let's assume that the machine is sorely designated
                (not given as a set, or a list)
                """
                # compatible machine list
                machine = operation.machine_list[0]
                # process time on the certain machine
                process_time = operation.process_time[0]
            else:  # if the compatible machine is given as integer
                machine = self.model['M' + str(operation.machine_list)]
                process_time = operation.process_time
            yield machine.availability.put('using')
            machine.add_reference(operation)

            # 3. Proceed & Record through console
            self.monitor.record(self.env.now, self.name, machine='M' + str(operation.machine_list),
                                part_name=part.name, event="Started")
            monitor_by_console(CONSOLE_MODE, self.env, part, OBJECT, "Started on")

            yield self.env.timeout(process_time)

            self.monitor.record(self.env.now, self.name, machine='M' + str(operation.machine_list),
                                part_name=part.name, event="Finished")
            monitor_by_console(CONSOLE_MODE, self.env, part, OBJECT, "Finished on")
            machine.util_time += process_time
            self.input_event.succeed()
            self.input_event = simpy.Event(self.env)

            # 4. Send(route) to the out_part queue for routing and update the machine availability
            yield self.out_part.put(part)
            yield machine.availability.get()

    def dispatch(self):
        while True:

            yield self.input_event
            if DISPATCH_MODE == 'FIFO':
                # # # Version 1 - FIFO Rule
                part_ready = yield self.in_part.get()
                yield self.part_ready.put(part_ready)
                # self.ready_event.succeed()
                # self.ready_event = simpy.Event(self.env)

            elif DISPATCH_MODE == 'Manual':
                # # Version 2 - Solution Rule
                num_scan = len(self.in_part.items)
                for i in range(num_scan):
                    if self.check_item():
                        part_ready = yield self.in_part.get(lambda x: x.part_type == self.machine_order[self.scheduled])
                        # print("Part ", part_ready.part_type, "is prepared")
                        # self.part_ready.put(part_ready)
                        yield self.part_ready.put(part_ready)
                        self.scheduled += 1

    def check_item(self):
        if CONSOLE_MODE & (self.name == 'Process4'):
            print('My Machine Order :',self.machine_order)
            print(self.name, ': Now I have', [i.part_type for i in self.in_part.items],'Jobs Waiting')

        for i, item in enumerate(self.in_part.items):
            if item.part_type == self.machine_order[self.scheduled]:
                if CONSOLE_MODE & (self.name == 'Process4'):
                    print('I gotta work on Job', self.machine_order[self.scheduled])
                return True

        return False

    def routing(self):
        while True:
            part = yield self.out_part.get()

            # update part status
            if part.step != 9:  # for operation 0,1,2,3 -> part.step = 1,2,3,4
            # if part.step != NUM_MACHINE - 1:  # for operation 0,1,2,3 -> part.step = 1,2,3,4
                part.step += 1
                part.op[part.step].requirements.succeed()
                next_process = self.model['Process' + str(part.op[part.step].process_type)]  # i.e. model['Process0']
                # The machine is not assigned yet (to be determined further)
                yield next_process.in_part.put(part)
                next_process.input_event.succeed()
                next_process.input_event = simpy.Event(self.env)
                part.loc = next_process.name
            else:
                self.model['Sink'].put(part)

