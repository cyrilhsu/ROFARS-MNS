# agent script for 'Resource Optimization for Facial Recognition Systems (ROFARS)' project
# author: Cyril Hsu @ UvA-MNS
# date: 23/02/2023

import numpy as np

class baselineAgent:

    def __init__(self, record_length=10):
        self.records = None
        self.record_length = record_length
        self.moving_avg = 0
        self.model = None

    def get_action(self, state):
        # add t orecord
        self.add_record(state)

        # random action
        #action = np.random.rand(len(state))

        # sum of historical records as scores (-1 is taken as 1 to encourage exploration)
        action = np.array([np.sum(np.abs(r)) for r in self.records])

        # normalization as the final step
        action = action/action.sum()
        return action

    def add_record(self, state):
        # init records w.r.t. the size of states
        if self.records is None:
            self.records = [[] for _ in range(len(state))]
        # insert records
        for r, s in zip(self.records, state):
            r.append(s)
            # discard oldest records
            if len(r) > self.record_length:
                r.pop(0)

    def clear_records(self):
        self.records = None

if __name__ == "__main__":
    agent = baselineAgent()
