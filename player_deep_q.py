from trainer_2048 import *
from deep_q import *
import numpy as np
import tkinter as tk

def run_game():
    root = tk.Tk()
    app = GUI(root)

    # agent = Q_Network(app, 16, Mean_Squared_Error_Loss, 4)
    # agent.add_layer(16, 16, ReLU)
    # agent.add_layer(16, 16, ReLU)
    # agent.add_layer(16, 4, Softmax)
    # agent.train(10)  # Train for 1000 episodes

    root.mainloop()


run_game()
