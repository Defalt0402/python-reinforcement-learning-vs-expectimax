import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

class Network:
    def __init__(self, inputNeurons, loss):
        self.hidden_layers = list()
        self.inputNeurons = inputNeurons
        self.loss = loss()
        self.alpha = 0.1
        self.stats_history = list()

    def save_model(self, name):
        model = {
            'inputNeurons': self.inputNeurons,
            'loss': self.loss.__class__,
            'hidden_layers': [(layer.weights, layer.biases, layer.activation.__class__) for layer in self.hidden_layers]
        }
        if not os.path.exists('model'):
            os.makedirs('model')
        joblib.dump(model, f"model/{name}.pkl")

    def load_model(self, name):
        model = joblib.load(f"model/{name}.pkl")
        net = Network(model['inputNeurons'], model['loss'])
        for weights, biases, activation in model['hidden_layers']:
            layer = Layer(0, 0, ReLU)
            layer.load_layer(weights, biases, activation)
            net.hidden_layers.append(layer)
        return net
            
    def add_layer(self, numInputs, neurons, activation):
        self.hidden_layers.append(Layer(numInputs, neurons, activation))

    def forward(self, inputs):
        self.inputs = inputs
        currentOutputs = inputs

        for layer in self.hidden_layers:
            layer.forward(currentOutputs)
            currentOutputs = layer.outputs

        return currentOutputs

    def partial_fit(self, inputs, y):
        # Calculate predictions and loss for reporting and use
        self.current_predictions = self.forward(inputs)
        self.current_loss = self.loss.calculate(self.current_predictions, y)
        
        # Backward pass
        gradient = self.loss.backward(self.current_predictions, y)
        for layer in reversed(self.hidden_layers):
            gradient = layer.backward(gradient, self.alpha)

    def train(self, inputs, y, epochs=1000, report=0, numReports=100):
        for i in range(epochs + 1):
            self.current_predictions = self.forward(inputs)
            self.current_loss = self.loss.calculate(self.current_predictions, y)
            
            # Backward pass
            gradient = self.loss.backward(self.current_predictions, y)
            for layer in reversed(self.hidden_layers):
                gradient = layer.backward(gradient, self.alpha)

            # Saves stats if report is 1
            # displays stats if report is 2
            # Does both if 3
            if report == 1:
                self.stats_history.append(self.get_stats(inputs, y))
            elif report == 2:
                if i % numReports == 0:
                    print(f"Epoch {i}, Loss: {self.current_loss}")
                    print(f"Stats {self.get_stats(inputs, y)}")
                    print(f"{self.current_predictions}\n")
            elif report == 3:
                self.stats_history.append(self.get_stats(inputs, y))
                if i % numReports == 0:
                    print(f"Epoch {i}, Loss: {self.current_loss}")
                    print(f"Stats {self.get_stats(inputs, y)}")
                    print(f"{self.current_predictions}\n")

    def get_stats(self, inputs, y):
        self.current_predictions = self.forward(inputs)
        yPred = np.argmax(self.current_predictions, axis=1)

        stats = {}
        stats["loss"] = self.loss.calculate(self.current_predictions, y)
        
        # Accuracy
        stats["accuracy"] = np.mean(yPred == y)
        
        # Precision
        true_positive = np.sum((yPred == y) & (yPred == 1))
        predicted_positive = np.sum(yPred == 1)
        stats["precision"] = true_positive / predicted_positive if predicted_positive > 0 else 0
        
        # Recall
        actual_positive = np.sum(y == 1)
        stats["recall"] = true_positive / actual_positive if actual_positive > 0 else 0
        
        # F1 Score
        precision = stats["precision"]
        recall = stats["recall"]
        stats["f1_score"] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return stats

    def plot_metrics(self, saveName=None):
        if len(self.stats_history) == 0:
            print("No data available to plot. Use report = 1 or report = 3 while training to generate data.")
            return

        lossVals = [stats["loss"] for stats in self.stats_history]
        accuracyVals = [stats["accuracy"] for stats in self.stats_history]
        precisionVals = [stats["precision"] for stats in self.stats_history]
        recallVals = [stats["recall"] for stats in self.stats_history]
        f1Vals = [stats["f1_score"] for stats in self.stats_history]
        epochs = range(0, len(self.stats_history))

        
        plt.figure(figsize=(10, 10))
        
        # Plot loss
        plt.subplot(3, 2, 1)
        plt.plot(epochs, lossVals, 'b-', label='Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(3, 2, 2)
        plt.plot(epochs, accuracyVals, 'r-', label='Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy over Epochs')
        plt.legend()
        
        # Plot precision
        plt.subplot(3, 2, 3)
        plt.plot(epochs, precisionVals, 'g-', label='Precision')
        plt.xlabel('Epochs')
        plt.ylabel('Precision')
        plt.title('Precision over Epochs')
        plt.legend()
        
        # Plot recall
        plt.subplot(3, 2, 4)
        plt.plot(epochs, recallVals, 'm-', label='Recall')
        plt.xlabel('Epochs')
        plt.ylabel('Recall')
        plt.title('Recall over Epochs')
        plt.legend()
        
        # Plot F1 score
        plt.subplot(3, 2, 5)
        plt.plot(epochs, f1Vals, 'c-', label='F1 Score')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.title('F1 Score over Epochs')
        plt.legend()
        
        plt.tight_layout()

        if saveName is not None:
            if not os.path.exists('graphs'):
                os.makedirs('graphs')
            plt.savefig(f"graphs/{saveName}.png")

        plt.show()
        

class Layer:
    def __init__(self, numInputs, neurons, activation):
        self.weights = np.random.rand(numInputs, neurons) * 0.01
        self.biases = np.zeros((1, neurons))
        self.activation = activation()

    def load_layer(self, weights, biases, activation):
        self.weights = weights
        self.biases = biases
        self.activation = activation()

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.biases
        self.activation.forward(self.outputs)
        self.outputs = self.activation.outputs

    def backward(self, gradient, learning_rate):
        self.activation.backward(gradient)
        gradient = self.activation.dinputs
        self.dweights = np.dot(self.inputs.T, gradient)
        self.dbiases = np.sum(gradient, axis=0, keepdims=True)
        
        self.dinputs = np.dot(gradient, self.weights.T)
        
        self.weights -= learning_rate * self.dweights
        self.biases -= learning_rate * self.dbiases
        
        return self.dinputs


class ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.maximum(0, inputs)

    def backward(self, gradient):
        self.dinputs = gradient.copy()
        self.dinputs[self.inputs <= 0] = 0


class Softmax:
    def forward(self, inputs):
        expVals = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.outputs = expVals / np.sum(expVals, axis=1, keepdims=True) 

    def backward(self, gradient):
        self.dinputs = gradient.copy() 

class Linear:
    def forward(self, inputs):
        self.outputs = inputs

    def backward(self, gradient):
        self.dinputs = gradient.copy()



class Loss:
    def calculate(self, yPred, y):
        return np.mean(self.forward(yPred, y))
    
class Cross_Entropy_Loss(Loss):
    def forward(self, yPred, y):
        numSamples = len(yPred)
        yPredClip = np.clip(yPred, 1e-8, 1-1e-8)

        if len(y.shape) == 1:
            confidence = yPredClip[range(numSamples), y]
        elif len(y.shape) == 2:
            confidence = np.sum(yPredClip*y, axis=1)
        
        return -np.log(confidence)

    def backward(self, yPred, y):
        numSamples = len(yPred)
        if len(y.shape) == 1:
            labels = y
        elif len(y.shape) == 2:
            labels = np.argmax(y, axis=1)
        
        self.dinputs = yPred.copy()
        self.dinputs[range(numSamples), labels] -= 1
        self.dinputs = self.dinputs / numSamples
        return self.dinputs
    
class Mean_Squared_Error_Loss(Loss):
    def calculate(self, yPred, y):
        return np.mean((yPred - y) ** 2)
    
    def backward(self, yPred, y):
        return 2 * (yPred - y) / y.size