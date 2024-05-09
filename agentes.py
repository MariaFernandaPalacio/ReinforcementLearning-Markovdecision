'''
Classes for implementing the learning methods.
'''
from random import randint, choice
import numpy as np
import random

class Agent :
    '''
    Defines the basic methods for the agent.
    '''

    def __init__(self, parameters:dict):
        self.parameters = parameters
        self.nS = self.parameters['nS']
        self.nA = self.parameters['nA']
        self.gamma = self.parameters['gamma']
        self.epsilon = self.parameters['epsilon']
        self.states = []
        self.actions = []
        self.rewards = [np.nan]
        self.dones = []
        self.policy = np.ones((self.nS, self.nA)) * 1/self.nA
        self.Q = np.zeros((self.nS, self.nA))

    def make_decision(self):
        '''
        Agent makes a decision according to its model.
        '''
        state = self.states[-1]
        weights = [self.policy[state, action] for action in range(self.nA)]
        action = random.choices(population = range(self.nA),\
                                weights = weights,\
                                k = 1)[0]
        return action

    def restart(self):
        '''
        Restarts the agent for a new trial.
        '''
        self.states = []
        self.actions = []
        self.rewards = [np.nan]

    def reset(self):
        '''
        Resets the agent for a new simulation.
        '''
        self.states = []
        self.actions = []
        self.rewards = []
        self.policy = np.ones((self.nS, self.nA)) * 1/self.nA
        self.Q = np.zeros((self.nS, self.nA))

    def max_Q(self, s):
        '''
        Determines the max Q value in state s
        '''
        return max([self.Q[s, a] for a in range(self.nA)])

    def argmaxQ(self, s):
        '''
        Determines the action with max Q value in state s
        '''
        maxQ = self.max_Q(s)
        opt_acts = [a for a in range(self.nA) if self.Q[s, a] == maxQ]
        return random.choice(opt_acts)

    def update_policy(self, s):
        opt_act = self.argmaxQ(s)
        prob_epsilon = lambda action: 1 - self.epsilon if action == opt_act else self.epsilon/(self.nA-1)
        self.policy[s] = [prob_epsilon(a) for a in range(self.nA)]

    def update(self, next_state, reward, done):
        '''
        Agent updates its model.
        TO BE DEFINED BY SUBCLASS
        '''
        pass

