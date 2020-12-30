import tkinter as tk
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import ImageTk, Image
import threading

sns.set(style="ticks")

pd.set_option('display.max_columns',1000)
pd.set_option('display.width', 10000000)
pd.set_option('display.max_rows', 100000)

class Stat_Data(tk.Frame):
    def __init__(self,parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.keys = self.controller.frames_list

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=12)
        for x in range(14):
            self.grid_rowconfigure(x, weight=1)
    def stat(self):
        if type(self.controller.df) !=str:
            for widget in self.winfo_children():
                widget.destroy()

            self.main_fr = tk.Frame(self,  relief='groove', borderwidth=5)
            self.main_fr.grid(row=0,column=0,rowspan=14,sticky='nsew')

            self.sec_fr = tk.Frame(self,relief='groove', borderwidth=5)
            self.sec_fr.grid(row=0,column=1,rowspan=14,sticky='nsew')

            self.main_fr.grid_columnconfigure(0, weight=1)

            self.sec_fr.grid_columnconfigure(0, weight=1)
            self.sec_fr.grid_rowconfigure(0, weight=1)
            #self.main_fr.grid_rowconfigure(0, weight=1)

            self.controller.show_frame(Stat_Data)

            self.l1 = tk.Button(self.main_fr, text="Head of data",relief='groove',command=self.head)
            self.l1.grid(row=0,column=0, sticky='nsew',pady=5)

            self.l2 = tk.Button(self.main_fr, text="Tail of data",relief='groove',command=self.tail)
            self.l2.grid(row=1,column=0, sticky='nsew',pady=5)

            self.l3 = tk.Button(self.main_fr, text="Info",relief='groove',command=self.info)
            self.l3.grid(row=2,column=0, sticky='nsew',pady=5)

            self.l4 = tk.Button(self.main_fr, text="Describe",relief='groove', command=self.describe)
            self.l4.grid(row=3,column=0, sticky='nsew',pady=5)

            self.l5 = tk.Button(self.main_fr, text="Correlation",relief='groove', command=self.corr)
            self.l5.grid(row=4,column=0, sticky='nsew',pady=5)

            self.l6 = tk.Button(self.main_fr, text="Type of Columns",relief='groove', command=self.types)
            self.l6.grid(row=5,column=0, sticky='nsew',pady=5)

            self.l7 = tk.Button(self.main_fr, text="NaN",relief='groove', command=self.NaN)
            self.l7.grid(row=6,column=0, sticky='nsew',pady=5)

            self.l8 = tk.Button(self.main_fr, text="Duplicated",relief='groove', command=self.Duplicated)
            self.l8.grid(row=7,column=0, sticky='nsew',pady=5)

            self.l8 = tk.Button(self.main_fr, text="Scatter Matrix",relief='groove',
                                command=threading.Thread(name="scatter", target=self.scatter_plot).start)
            self.l8.grid(row=8,column=0, sticky='nsew',pady=5)

            self.l9 = tk.Button(self.main_fr, text="Histograms",relief='groove', command=self.Histogram)
            self.l9.grid(row=9,column=0, sticky='nsew',pady=5)
        else:
            self.controller.error()


    def head(self):
        for widget in self.sec_fr.winfo_children():
            widget.destroy()
        self.ab = tk.Tk()
        self.ab.wm_title("Number of rows")
        fr = tk.Frame(self.ab)
        fr.grid()
        self.l1 = tk.Label(fr, text="Please indicate how many rows would you like to display:")
        self.l1.grid(row=0, column=0, sticky='w')
        self.e1 = tk.Entry(fr, width=4)
        self.e1.insert(1, 5)
        self.e1.grid(row=0, column=1, sticky='e')
        self.b1 = tk.Button(fr, text="Confirm", command=self.head_display)
        self.b1.grid(row=10, column=0)
        self.ab.mainloop()

    def head_display(self):
        for widget in self.sec_fr.winfo_children():
            widget.destroy()

        scroll = tk.Scrollbar(self)
        scrollx = tk.Scrollbar(self, orient=tk.HORIZONTAL)
        scroll.grid(row=0, column=125, rowspan=50, sticky='ns')
        scrollx.grid(row=15, column=1, columnspan=160, sticky='ew')

        self.t1 = tk.Text(self.sec_fr, borderwidth=0,wrap=tk.NONE, yscrollcommand=scroll.set, xscrollcommand=scrollx.set)
        self.t1.insert(tk.END,self.controller.df.head(int(self.e1.get())))
        self.t1.grid(row=0,column=0,sticky='nsew')
        scroll.config(command=self.t1.yview)
        scrollx.config(command=self.t1.xview)
        self.ab.destroy()

    def tail(self):
        for widget in self.sec_fr.winfo_children():
            widget.destroy()
        self.ab = tk.Tk()
        self.ab.wm_title("Number of rows")
        fr = tk.Frame(self.ab)
        fr.grid()
        self.l1 = tk.Label(fr, text="Please indicate how many rows would you like to display:")
        self.l1.grid(row=0, column=0, sticky='w')
        self.e1 = tk.Entry(fr, width=4)
        self.e1.insert(1, 5)
        self.e1.grid(row=0, column=1, sticky='e')
        self.b1 = tk.Button(fr, text="Confirm", command=self.tail_display)
        self.b1.grid(row=10, column=0)
        self.ab.mainloop()

    def tail_display(self):
        for widget in self.sec_fr.winfo_children():
            widget.destroy()
        scroll = tk.Scrollbar(self)
        scrollx = tk.Scrollbar(self, orient=tk.HORIZONTAL)
        scroll.grid(row=0, column=125, rowspan=50, sticky='ns')
        scrollx.grid(row=15, column=1, columnspan=160, sticky='ew')

        self.t1 = tk.Text(self.sec_fr, borderwidth=0,wrap=tk.NONE, yscrollcommand=scroll.set, xscrollcommand=scrollx.set)
        self.t1.insert(tk.END,self.controller.df.tail(int(self.e1.get())))
        self.t1.grid(row=0,column=0,sticky='nsew')
        scroll.config(command=self.t1.yview)
        scrollx.config(command=self.t1.xview)
        self.ab.destroy()

    def info(self):
        for widget in self.sec_fr.winfo_children():
            widget.destroy()
        scroll = tk.Scrollbar(self)
        scrollx = tk.Scrollbar(self, orient=tk.HORIZONTAL)
        scroll.grid(row=0, column=125, rowspan=50, sticky='ns')
        scrollx.grid(row=15, column=1, columnspan=160, sticky='ew')

        self.t1 = tk.Text(self.sec_fr, borderwidth=0,wrap=tk.NONE, yscrollcommand=scroll.set, xscrollcommand=scrollx.set)

        with open("buf.txt",'w') as f:
            self.controller.df.info(buf=f)

        self.t1.insert(tk.END, open("buf.txt").readlines())
        self.t1.grid(row=0,column=0,sticky='nsew')
        scroll.config(command=self.t1.yview)
        scrollx.config(command=self.t1.xview)

    def describe(self):
        for widget in self.sec_fr.winfo_children():
            widget.destroy()
        scroll = tk.Scrollbar(self)
        scrollx = tk.Scrollbar(self, orient=tk.HORIZONTAL)
        scroll.grid(row=0, column=125, rowspan=50, sticky='ns')
        scrollx.grid(row=15, column=1, columnspan=160, sticky='ew')

        self.t1 = tk.Text(self.sec_fr, borderwidth=0,wrap=tk.NONE, yscrollcommand=scroll.set, xscrollcommand=scrollx.set)
        self.t1.insert(tk.END,self.controller.df.describe())
        self.t1.grid(row=0,column=0,sticky='nsew')
        scroll.config(command=self.t1.yview)
        scrollx.config(command=self.t1.xview)

    def corr(self):
        for widget in self.sec_fr.winfo_children():
            widget.destroy()
        scroll = tk.Scrollbar(self)
        scrollx = tk.Scrollbar(self, orient=tk.HORIZONTAL)
        scroll.grid(row=0, column=125, rowspan=50, sticky='ns')
        scrollx.grid(row=15, column=1, columnspan=160, sticky='ew')

        self.t1 = tk.Text(self.sec_fr, borderwidth=0,wrap=tk.NONE, yscrollcommand=scroll.set, xscrollcommand=scrollx.set)
        self.t1.insert(tk.END,self.controller.df.corr())
        self.t1.grid(row=0,column=0,sticky='nsew')
        scroll.config(command=self.t1.yview)
        scrollx.config(command=self.t1.xview)

    def types(self):
        for widget in self.sec_fr.winfo_children():
            widget.destroy()
        scroll = tk.Scrollbar(self)
        scrollx = tk.Scrollbar(self, orient=tk.HORIZONTAL)
        scroll.grid(row=0, column=125, rowspan=50, sticky='ns')
        scrollx.grid(row=15, column=1, columnspan=160, sticky='ew')

        self.t1 = tk.Text(self.sec_fr, borderwidth=0,wrap=tk.NONE, yscrollcommand=scroll.set, xscrollcommand=scrollx.set)
        self.t1.insert(tk.END,self.controller.df.dtypes)
        self.t1.grid(row=0,column=0,sticky='nsew')
        scroll.config(command=self.t1.yview)
        scrollx.config(command=self.t1.xview)

    def NaN(self):
        for widget in self.sec_fr.winfo_children():
            widget.destroy()
        scroll = tk.Scrollbar(self)
        scrollx = tk.Scrollbar(self, orient=tk.HORIZONTAL)
        scroll.grid(row=0, column=125, rowspan=50, sticky='ns')
        scrollx.grid(row=15, column=1, columnspan=160, sticky='ew')

        self.t1 = tk.Text(self.sec_fr, borderwidth=0,wrap=tk.NONE, yscrollcommand=scroll.set, xscrollcommand=scrollx.set)
        self.t1.insert(tk.END,self.controller.df[self.controller.df.isna().any(axis=1)])
        self.t1.grid(row=0,column=0,sticky='nsew')
        scroll.config(command=self.t1.yview)
        scrollx.config(command=self.t1.xview)

    def Duplicated(self):
        for widget in self.sec_fr.winfo_children():
            widget.destroy()
        scroll = tk.Scrollbar(self)
        scrollx = tk.Scrollbar(self, orient=tk.HORIZONTAL)
        scroll.grid(row=0, column=125, rowspan=50, sticky='ns')
        scrollx.grid(row=15, column=1, columnspan=160, sticky='ew')

        self.t1 = tk.Text(self.sec_fr, borderwidth=0,wrap=tk.NONE, yscrollcommand=scroll.set, xscrollcommand=scrollx.set)
        self.t1.insert(tk.END,self.controller.df[self.controller.df.duplicated()==True])
        self.t1.grid(row=0,column=0,sticky='nsew')
        scroll.config(command=self.t1.yview)
        scrollx.config(command=self.t1.xview)

    def scatter_plot(self):

        for widget in self.sec_fr.winfo_children():
            widget.destroy()
        self.l8['state'] = 'disable'
        canvas = tk.Canvas(self.sec_fr)
        canvas.grid(row=0,column=0,sticky='nsew')

        yScrollbar = tk.Scrollbar(self)
        xScrollbar = tk.Scrollbar(self, orient='horizontal')
        yScrollbar.grid(row=0, column=125, rowspan=50, sticky='ns')
        xScrollbar.grid(row=15, column=1, columnspan=160, sticky='ew')
        canvas.config(yscrollcommand=yScrollbar.set)
        canvas.config(xscrollcommand=xScrollbar.set)
        yScrollbar.config(command=canvas.yview)
        xScrollbar.config(command=canvas.xview)
        self.f = plt.figure(dpi=100, figsize=(25, 15))
        self.axes = self.f.subplots()
        ab = pd.plotting.scatter_matrix(self.controller.df, ax=self.axes, s=8,diagonal='kde')
        for ax in ab.ravel():
            ax.set_xlabel(ax.get_xlabel(), fontsize=7, rotation=45)
            ax.set_ylabel(ax.get_ylabel(), fontsize=7, rotation=0, labelpad=40)

        plt.tight_layout()
        self.f.savefig("scatter_matrix.png")
        self.im = ImageTk.PhotoImage(Image.open("scatter_matrix.png"))
        self.imgtag = canvas.create_image(0, 0, anchor="nw", image=self.im)
        canvas.config(scrollregion=canvas.bbox(tk.ALL))
        self.l8['state'] = 'normal'

    def Histogram(self):
        for widget in self.sec_fr.winfo_children():
            widget.destroy()

        yScrollbar = tk.Scrollbar(self)
        xScrollbar = tk.Scrollbar(self, orient='horizontal')
        yScrollbar.grid(row=0, column=125, rowspan=50, sticky='ns')
        xScrollbar.grid(row=15, column=1, columnspan=160, sticky='ew')
        self.f, self.axes = plt.subplots()

        self.canvas = FigureCanvasTkAgg(self.f, self.sec_fr)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.draw()
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.sec_fr)
        self.toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.axes.clear()
        self.controller.df.hist(ax=self.axes)

        plt.tight_layout()
        self.canvas.draw()
        self.toolbar.update()