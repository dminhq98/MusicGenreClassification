import SongClassifier
from tkinter import *
from math import *

import tkinter as tk


class SampleApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("Music Classifier")  # Application Name
        self.geometry("400x200")  # Application Size
        self.entry = tk.Entry(self,bd=5,width=40)
        self.button = tk.Button(self, text="Predict Genre", command=self.on_button)
        self.button.pack(side='bottom')
        self.entry.pack()
        self.labels = []

    def on_button(self):
        out = SongClassifier.test([self.entry.get()])
        # self.button.destroy()
        for label in self.labels:
            label.destroy()
        label = Label(self, text=str(out),height=20,fg="red",font=("Helvetica", 20))
        self.labels.append(label)
        label.pack()

    def button(self):
        pass


app = SampleApp()
app.mainloop()
