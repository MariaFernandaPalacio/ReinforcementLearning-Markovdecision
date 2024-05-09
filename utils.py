'''
Helper functions to gather, process and visualize data
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from seaborn import lineplot, histplot, heatmap, color_palette
from tqdm import tqdm
from copy import deepcopy
from time import sleep
from IPython.display import clear_output, display
import json

class Episode :
    '''
    Runs the environment for a number of rounds and keeps tally of everything.
    '''

    def __init__(self, environment, agent, model_name:str, num_rounds:int, id:int=0):
        self.environment = environment
        self.agent = agent
        self.model_name = model_name
        self.num_rounds = num_rounds
        self.done = False
        self.T = 1
        self.id = id
        self.sleep_time = 0.1
        state = self.environment.reset()
        # Some environments from gymnastics have weird states as tuples.
        # If so, keep only first element from tuple.
        if isinstance(state, tuple):
            state = state[0]
        self.initial_state = state
        if agent is not None:
            self.agent.restart()
            self.agent.states.append(state)

    def play_round(self, verbose:int=0):
        '''
        Plays one round of the game.
        Input:
            - verbose, to print information.
                0: no information
                1: only number of simulation
                2: simulation information
                3: simulation and episode information
                4: simulation, episode and round information
        '''
        # Ask agent to make a decision
        action = self.agent.make_decision()
        self.agent.actions.append(action)
        # Runs the environment and obtains the next_state, reward, done
        result = self.environment.step(action)
        next_state, reward, done = result[0], result[1], result[2]
        # Some environments from gymnastics have weird states as tuples.
        # If so, keep only first element from tuple.
        if isinstance(next_state, tuple):
            next_state = next_state[0]
        # Print info
        if verbose > 3:
            state = self.agent.states[-1]
            print(f'\tThe state is => {state}')
            print(f'\tAgent takes action => {action}')
            print(f'\tThe state obtained is => {next_state}')
            print(f'\tThe reward obtained is => {reward}')
            print(f'\tEnvironment is finished? => {done}')
            print(f'\t{self.environment.agente}, {self.environment.dir_agente}')
        self.agent.states.append(next_state)
        self.agent.rewards.append(reward)
        self.agent.dones.append(done)
        self.T += 1
        self.done = done

    def run(self, verbose:int=0):
        '''
        Plays the specified number of rounds.
        '''
        for round in range(self.num_rounds):
            if not self.done:
                if verbose > 2:
                    print('\n' + '-'*10 + f'Round {round}' + '-'*10 + '\n')
                self.play_round(verbose=verbose)                
            else:
                break
        return self.to_pandas()

    def to_pandas(self) -> pd.DataFrame:
        '''
        Creates a pandas dataframe with the information from the current objects.
        Output:
            - pandas dataframe with the following variables:           
                Variables:
                    * episode: a unique identifier for the episode
                    * round: the round number
                    * action: the player's action
                    * reward: the player's reward
                    * done: whether the environment is active or not
                    * model: the model's name
        '''
        # Include las item in actions list
        self.agent.actions.append(np.nan)
        # n1 = len(self.agent.states)
        # n2 = len(self.agent.actions)
        # n3 = len(self.agent.rewards)
        # n4 = len(self.agent.dones)
        # print(n1, n2, n3, self.T)
        data = {}
        data["episode"] = []
        data["round"] = []
        data["state"] = []
        data["action"] = []
        data["reward"] = []
        data["done"] = []
        for r in range(self.T):
            data["episode"].append(self.id)
            data["round"].append(r)
            data["state"].append(self.agent.states[r])
            data["action"].append(self.agent.actions[r])
            data["reward"].append(self.agent.rewards[r])
            data["done"].append(self.agent.dones[r])
        df = pd.DataFrame.from_dict(data)        
        df["model"] = self.model_name
        return df

    def reset(self):
        '''
        Reset the episode. This entails:
            reset the environment
            restart the agent 
                  (new states, actions and rewards, 
                   but keep what was learned)
        '''
        state = self.environment.reset()
        # Some environments from gymnastics have weird states as tuples.
        # If so, keep only first element from tuple.
        if isinstance(state, tuple):
            state = state[0]
        self.initial_state = state
        self.agent.restart()
        self.agent.states.append(state)
        self.T = 1
        self.done = False

    def renderize(self):
        '''
        Plays the specified number of rounds.
        '''
        img = plt.imshow(np.array([[0, 0], [0, 0]])) # only call this once
        for round in range(self.num_rounds):
            if not self.done:
                self.play_round(verbose=0)                
                im = self.environment.render()
                if isinstance(im, np.ndarray):
                    img.set_data(self.environment.render())
                    display(plt.gcf())
                sleep(self.sleep_time)
                clear_output(wait=True)
            else:
                break

    def simulate(self, num_episodes:int=1, file:str=None, verbose:int=0):
        '''
        Runs the specified number of episodes for the given number of rounds.
        Input:
            - num_episodes, int with the number of episodes.
            - file, string with the name of file to save the data on.
            - verbose, to print information.
                0: no information
                1: only number of simulation
                2: simulation information
                3: simulation and episode information
                4: simulation, episode and round information
        Output:
            - Pandas dataframe with the following variables:

                Variables:
                    * id_sim: a unique identifier for the simulation
                    * round: the round number
                    * action: the player's action
                    * reward: the player's reward
                    * done: whether the environment is active or not
                    * model: the model's name
        '''
        # Create the list of dataframes
        data_frames = []
        # Run the number of episodes
        for ep in range(num_episodes):
            if verbose > 1:
                print('\n' + '='*10 + f'Episode {ep}' + '='*10 + '\n')
            # Reset the episode
            self.reset()
            self.id = ep
            # Run the episode
            df = self.run(verbose=verbose)
            # print(self.agent.Q)
            # Include episode in list of dataframes
            data_frames.append(df)
        # Concatenate dataframes
        data = pd.concat(data_frames, ignore_index=True)
        if file is not None:
            data.to_csv(file)
        return data

class PlotGridValues :
    
    def __init__(self, shape:tuple, action_dict:dict):
        assert(len(shape) == 2)
        self.shape = shape
        self.action_dict = action_dict
        self.nA = len(action_dict.keys())
        
    def plot_policy(self, policy, V=None, ax=None):
        assert(self.shape == policy.shape)
        annotations = np.vectorize(self.action_dict.get)(policy)
        if V is None:
            values = np.zeros(self.shape)
        else:
            assert(self.shape == V.shape)
            values = V
        if ax is None:
            heatmap(
                values,
                annot=annotations,
                cbar=False,
                fmt="",
                cmap=color_palette("Blues", as_cmap=True),
                linewidths=1.7,
                linecolor="black",
                xticklabels=[],
                yticklabels=[],
                annot_kws={"fontsize": "xx-large"},
            ).set(title="Policy: action per state")
            plt.plot()
        else:
            heatmap(
                values,
                annot=annotations,
                cbar=False,
                fmt="",
                cmap=color_palette("Blues", as_cmap=True),
                linewidths=1.7,
                linecolor="black",
                xticklabels=[],
                yticklabels=[],
                annot_kws={"fontsize": "xx-large"},
                ax = ax
            ).set(title="Policy: action per state")

    def plot_V_values(self, V, ax=None):
        assert(self.shape == V.shape)
        if ax is None:
            heatmap(
                V,
                annot=True,
                fmt="",
                cmap=color_palette("Blues", as_cmap=True),
                linewidths=0.7,
                linecolor="black",
                xticklabels=[],
                yticklabels=[],
                annot_kws={"fontsize": "x-large"},
            ).set(title="V-values")
        else:
            heatmap(
                V,
                annot=True,
                fmt="",
                cmap=color_palette("Blues", as_cmap=True),
                linewidths=0.7,
                linecolor="black",
                xticklabels=[],
                yticklabels=[],
                annot_kws={"fontsize": "x-large"},
                ax = ax
            ).set(title="V-values")
