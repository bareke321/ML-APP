from tkinter import messagebox
from tkinter import filedialog
import tkinter as tk


class Show_Data(tk.Frame):
    def __init__(self, parent, controller):
        self.controller = controller
        super().__init__(parent)
        self.keys = self.controller.frames_list

    def show_df(self):
        for widget in self.winfo_children():
            widget.destroy()

        if type(self.controller.df)!=str:
            self.controller.show_frame(Show_Data)
            self.df = self.controller.df

            scrollbar = tk.Scrollbar(self)
            scrollbar.pack(side='right', fill='y')
            scrolbarX = tk.Scrollbar(self, orient='horizontal')
            multi = tk.Text(self, borderwidth=1, relief="solid",wrap=tk.NONE, yscrollcommand=scrollbar.set, xscrollcommand=scrolbarX.set)
            multi.insert(tk.END, self.df)
            multi.pack(side='top', fill='both', expand=True)
            scrollbar.config(command=multi.yview)
            scrolbarX.config(command=multi.xview)
            scrolbarX.pack(side="bottom", fill='x')
        else:
            self.controller.error()