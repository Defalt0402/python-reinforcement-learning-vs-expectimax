from neural_net import *
from collections import deque
import random
import tkinter as tk
import numpy as np

class Q_Network(Network):
    def __init__(self, game, inputNeurons, loss, numActions, bufferSize=10000, batchSize=64, gamma=0.99, epsilon=1.0, minEpsilon=0.1, epsilonDecay=0.999, alpha=0.001):
        super().__init__(inputNeurons, loss)
        self.game = game()
        self.actionSpace = [i for i in range(numActions)]
        self.target_network = Network(inputNeurons, loss)
        self.target_network.hidden_layers = self.hidden_layers.copy()  # Copy the structure of the Q-network
        self.bufferSize = bufferSize
        self.buffer = deque(maxlen=bufferSize)
        self.batch_size = batchSize
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = minEpsilon
        self.epsilon_decay = epsilonDecay
        self.alpha = alpha
        self.score_history = list()

    def save_model(self, name):
        model = {
            'game': self.game.__class__,
            'numActions': len(self.actionSpace),
            'inputNeurons': self.inputNeurons,
            'loss': self.loss.__class__,
            'hidden_layers': [(layer.weights, layer.biases, layer.activation.__class__) for layer in self.hidden_layers],
            'target_hidden_layers': [(layer.weights, layer.biases, layer.activation.__class__) for layer in self.target_network.hidden_layers],
            'bufferSize': self.bufferSize,  # Convert deque to list for saving
            'batchSize': self.batch_size,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'minEpsilon': self.epsilon_min,
            'epsilonDecay': self.epsilon_decay,
            'alpha': self.alpha
        }
        if not os.path.exists('model'):
            os.makedirs('model')
        joblib.dump(model, f"model/{name}.pkl")


    def load_model(self, name):
        model = joblib.load(f"model/{name}.pkl")
        net = Q_Network(model["game"], model["inputNeurons"], model["loss"], model["numActions"], model["batchSize"], model["gamma"],
                        model["epsilon"], model["minEpsilon"], model["epsilonDecay"], model["alpha"])
        for weights, biases, activation in model['hidden_layers']:
            layer = Layer(0, 0, activation)
            layer.load_layer(weights, biases)
            net.hidden_layers.append(layer)
        for weights, biases, activation in model['target_hidden_layers']:
            layer = Layer(0, 0, activation)
            layer.load_layer(weights, biases)
            net.target_network.hidden_layers.append(layer)
        return net

    def one_hot_encode(self, possibleVals, val):
        oneHotVector = np.zeros(len(possibleVals))

        if val in possibleVals:
            index = possibleVals.index(val)
            oneHotVector[index] = 1

        return oneHotVector
    
    def one_hot_encode_board(self, board):
        possibleVals = [2**i for i in range(1, 17)] # array of powers of 2 from 2 to 64000
        oneHotVector = np.array([self.one_hot_encode(possibleVals, val) for val in board.flatten()]).flatten().reshape(1, -1)
        return oneHotVector
        
    def add_layer(self, numInputs, neurons, activation):
        self.hidden_layers.append(Layer(numInputs, neurons, activation))
        self.target_network.hidden_layers = self.hidden_layers.copy()

    def store_experience(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.actionSpace)
        oneHotState = self.one_hot_encode_board(state)
        q_values = self.forward(oneHotState)
        return np.argmax(q_values)
    
    def replay(self):
        if len(self.buffer) < self.batch_size:
            return

        minibatch = random.sample(self.buffer, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            oneHotState = self.one_hot_encode_board(state)
            target = self.forward(oneHotState)
            if done:
                target[0][action] = reward - 1000
            else:
                oneHotNextState = self.one_hot_encode_board(next_state)
                t = self.target_network.forward(oneHotNextState)
                target[0][action] = reward + self.gamma * np.amax(t)
            self.partial_fit(oneHotState, target)

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            
    def update_target_network(self):
        self.target_network.hidden_layers = self.hidden_layers.copy()

    def train(self, episodes, gui_callback=None):
        for episode in range(episodes):
            self.game.reset()
            state = self.game.board
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.step(action)
                self.store_experience(state, action, reward, next_state, done)
                state = next_state
                self.replay()

                if gui_callback is not None:
                    gui_callback()
            self.update_target_network()
            self.update_epsilon()
            self.score_history.append(self.game.score)
            print(f"Episode {episode + 1}/{episodes}, Score: {self.game.score}, Epsilon: {self.epsilon}")

        self.plot_metrics("2048_agent_no_score")
        self.save_model("2048_agent_no_score")
            
    def step(self, action):
        if action == 0:
            moved = self.game.slide_left()
        elif action == 1:
            moved = self.game.slide_right()
        elif action == 2:
            moved = self.game.slide_up()
        elif action == 3:
            moved = self.game.slide_down()
        
        reward = self.game.evaluate_board()
        next_state = self.game.board
        done = self.game.is_game_over()
        return next_state, reward, done

    def plot_metrics(self, saveName=None):
        if len(self.score_history) == 0:
            print("No data available to plot.")
            return

        episodes = range(1, len(self.score_history) + 1)
        
        # Compute moving average of scores
        windowSize = 50
        if len(self.score_history) >= windowSize:
            movingAverage = np.convolve(self.score_history, np.ones(windowSize) / windowSize, mode='valid')
        else:
            movingAverage = []

        plt.figure(figsize=(15, 8))

        # Plot scores
        plt.subplot(1, 2, 1)
        plt.plot(episodes, self.score_history, label='Score per Episode', color='b')
        plt.xlabel('Episodes')
        plt.ylabel('Score')
        plt.title('Score per Episode')
        plt.legend()

        # Plot moving average of scores
        plt.subplot(1, 2, 2)
        plt.plot(range(windowSize, len(self.score_history) + 1), movingAverage, label='Moving Average of Scores', color='r')
        plt.xlabel('Episodes')
        plt.ylabel('Moving Average Score')
        plt.title(f'Moving Average (window size = {windowSize}) of Scores')
        plt.legend()

        plt.tight_layout()

        if saveName is not None:
            if not os.path.exists('graphs'):
                os.makedirs('graphs')
            plt.savefig(f"graphs/{saveName}.png")

        plt.show()

