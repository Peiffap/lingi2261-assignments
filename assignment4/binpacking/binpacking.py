#! /usr/bin/env python3
"""NAMES OF THE AUTHOR(S): Gaël Aglin     <gael.aglin@uclouvain.be>
                           Martin Braquet <martin.braquet@student.uclouvain.be>
                           Gilles Peiffer <gilles.peiffer@student.uclouvain.be>
"""
from search import *
import sys
import random
import time


class BinPacking(Problem):

    def successor(self, state):
        items = list(state.items.items())
        bins = state.bins
        c = state.capacity
        
        def find_bin_by_item(key):
            '''
            Find the bin in which a given item sits.
            '''
            ind = 0
            for b in state.bins:
                if key in b:
                    return b, ind
                ind += 1
        
        def deepcopy_list_of_dicts(lod):
            '''
            Deepcopies a list of dictionaries.
            '''
            newlod = []
            for d in lod:
                newlod.append(d.copy())
            return newlod
        
        # Generate "swap two items"-type moves.
        for (item1, item2) in [(a, b) for a in items for b in items]:
            val1, key1 = item1[1], item1[0]
            val2, key2 = item2[1], item2[0]
            bin1, ind1 = find_bin_by_item(key1)
            bin2, ind2 = find_bin_by_item(key2)
            if val1 <= val2 and ind1 != ind2 and state.can_fit(bin1, val2-val1) and state.can_fit(bin2, val1-val2):
                # Check that
                # -  the item pairs appear only once;
                # -  the bins are different; and
                # -  both bins can accomodate the swap.
                # If so, apply the swap and yield the resulting state.
                newbins = deepcopy_list_of_dicts(bins)
                del newbins[ind1][key1]
                newbins[ind1][key2] = val2
                del newbins[ind2][key2]
                newbins[ind2][key1] = val1
                s = State(c, state.items, newbins)
                yield ("swpitit", s) # Yield the new state and indicate item-item swap.
        
        # Generate "swap item and blank space"-type moves.
        for (item, b) in [(a, b) for a in items for b in list(range(len(bins)))]:
            val, key = item[1], item[0]
            _, ind = find_bin_by_item(key)
            if ind != b and state.can_fit(bins[b], val):
                # Check that
                # -  the item's current bin and swapped bin are different
                # -  the new bin can accomodate the swap.
                # If so, apply the swap and yield the resulting state.
                newbins = deepcopy_list_of_dicts(bins)
                del newbins[ind][key]
                newbins[b][key] = val
                try:
                    newbins.remove({}) # Remove empty bins if any.
                except ValueError:
                    pass  # Skip.
                s = State(c, state.items, newbins)
                yield ("swpitbs", s) # Yield the new state and indicate item-blank space swap.
    

    def fitness(self, state):
        """
        :param state:
        :return: fitness value of the state in parameter
        """
        s = 0
        for bin in state.bins:
            s += (sum(list(bin.values()))/state.capacity)**2
        return s/len(state.bins) - 1
    
    def value(self, state):
        return self.fitness(state)


class State:

    def __init__(self, capacity, items, bins="def"):
        self.capacity = capacity
        self.items = items
        if bins == "def":
            self.bins = self.build_init()
        else:
            self.bins = bins

    # an init state building is provided here but you can change it at will
    def build_init(self):
        init = []
        for ind, size in self.items.items():
            if len(init) == 0 or not self.can_fit(init[-1], size):
                init.append({ind: size})
            else:
                if self.can_fit(init[-1], size):
                    init[-1][ind] = size
        return init

    def can_fit(self, bin, itemsize):
        return sum(list(bin.values())) + itemsize <= self.capacity

    def __str__(self):
        s = ''
        for i in range(len(self.bins)):
            s += ' '.join(list(self.bins[i].keys())) + '\n'
        return s


def read_instance(instanceFile):
    file = open(instanceFile)
    capacitiy = int(file.readline().split(' ')[-1])
    items = {}
    line = file.readline()
    while line:
        items[line.split(' ')[0]] = int(line.split(' ')[1])
        line = file.readline()
    return capacitiy, items

# Attention : Depending of the objective function you use, your goal can be to maximize or to minimize it
def maxvalue(problem, limit=100, callback=None):
    current = LSNode(problem, problem.initial, 0)
    best = current
    bfit = problem.fitness(current.state)
    
    for step in range(limit):
        nb = list(current.expand())
        neighbours = list(enumerate(nb))
        values = sorted(map(lambda x: (x[0], problem.fitness(x[1].state)), neighbours), key=lambda x: x[1], reverse=True)
        current = nb[values[0][0]]
        if values[0][1] > bfit:
            best = LSNode(problem, current.state, step+1)
            bfit = values[0][1]

    return best

# Attention : Depending of the objective function you use, your goal can be to maximize or to minimize it
def randomized_maxvalue(problem, limit=100, callback=None):
    current = LSNode(problem, problem.initial, 0)
    best = current
    bfit = problem.fitness(current.state)
    
    for step in range(limit):
        nb = list(current.expand())
        neighbours = list(enumerate(nb))
        values = sorted(map(lambda x: (x[0], problem.fitness(x[1].state)), neighbours), key=lambda x: x[1], reverse=True)
        top5 = values[0:5]
        current = nb[random.choice(top5)[0]]
        if values[0][1] > bfit:
            best = LSNode(problem, current.state, step+1)
            bfit = values[0][1]

    return best

def numerical_experiment():
    def fmat(i, n):
        return '{:g}'.format(float('{:.{p}g}'.format(i, p=n)))
    step_limit = 100
    s = ''
    for i in range(10):
        info = read_instance("instances/i" + '{:02d}'.format(i+1) + ".txt")
        init_state = State(info[0], info[1])
        bp_problem = BinPacking(init_state)
        startTime = time.perf_counter()
        node = maxvalue(bp_problem, step_limit)
        endTime = time.perf_counter()
        state = node.state
        s += str(i+1) + " & " + fmat(1000*(endTime-startTime), 3) + " & \\(" + fmat(bp_problem.fitness(state), 6) + "\\) & " + str(node.step) + " & "
        
        times = []
        fitnesses = []
        steps = []
        for j in range(10):
            startTime = time.perf_counter()
            node = randomized_maxvalue(bp_problem, step_limit)
            endTime = time.perf_counter()
            times.append(endTime-startTime)
            state = node.state
            fitnesses.append(bp_problem.fitness(state))
            steps.append(node.step)
        t = 100*sum(times)
        f = sum(fitnesses)/len(fitnesses)
        step = sum(steps)/len(steps)
        s += fmat(t, 3) + " & \\(" + fmat(f, 6) + "\\) & " + fmat(step, 3) + " & "
        
        times = []
        fitnesses = []
        steps = []
        for j in range(10):
            startTime = time.perf_counter()
            node = random_walk(bp_problem, step_limit)
            endTime = time.perf_counter()
            times.append(endTime-startTime)
            state = node.state
            fitnesses.append(bp_problem.fitness(state))
            steps.append(node.step)
        t = 100*sum(times)
        f = sum(fitnesses)/len(fitnesses)
        step = sum(steps)/len(steps)
        s += fmat(t, 3) + " & \\(" + fmat(f, 6) + "\\) & " + fmat(step, 3) + " \\\\\n"
    print(s)
#####################
#       Launch      #
#####################
if __name__ == '__main__':
    info = read_instance(sys.argv[1])
    init_state = State(info[0], info[1])
    bp_problem = BinPacking(init_state)
    step_limit = 100
    #node = maxvalue(bp_problem, step_limit)
    node = randomized_maxvalue(bp_problem, step_limit)
    state = node.state
    print(state)
    # numerical_experiment()