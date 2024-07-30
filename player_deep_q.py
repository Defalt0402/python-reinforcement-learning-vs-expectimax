from trainer_2048 import *
from deep_q import *
import numpy as np
import tkinter as tk

def run_game():
    root = tk.Tk()
    app = GUI(root)

    root.mainloop()


run_game()
