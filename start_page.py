import tkinter as tk


class Start_Page(tk.Frame):
    def __init__(self, parent, controller, ):
        super().__init__(parent, relief="groove", borderwidth=3)

        self.controller = controller
        self.Label = tk.Label(self, text="\n\nGeofizyka 2020",font=("Times New Roman",80))
        self.Label.pack(side='top')




