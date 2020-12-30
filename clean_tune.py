import tkinter as tk
import matplotlib
from tkinter import messagebox
import pandas as pd
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set(style="ticks")

class Clean_Tune(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg='white')
        self.controller = controller
        self.keys = self.controller.frames_list

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

    def clean(self):
        for widget in self.winfo_children():
            widget.destroy()
        if type(self.controller.df) != str:
            self.controller.show_frame(Clean_Tune)

            self.main_fr = tk.Frame(self, relief='groove', borderwidth=5)
            self.main_fr.grid(row=0, column=0, sticky='nsew')
            self.main_fr.grid_columnconfigure(0, weight=1)
            self.main_fr.grid_rowconfigure(0, weight=1)
            self.main_fr.grid_rowconfigure(1, weight=1)


            self.sec_fr = tk.Frame(self, relief='groove', borderwidth=5)
            self.sec_fr.grid(row=0, column=1,sticky='nsew')
            self.sec_fr.grid_columnconfigure(0, weight=1)
            self.sec_fr.grid_rowconfigure(0, weight=1)

            scroll = tk.Scrollbar(self)
            scrollx = tk.Scrollbar(self, orient=tk.HORIZONTAL)
            scroll.grid(row=0, column=125, rowspan=50, sticky='ns')
            scrollx.grid(row=15, column=1, columnspan=160, sticky='ew')

            self.txt = tk.Text(self.sec_fr, borderwidth=0,wrap=tk.NONE, yscrollcommand=scroll.set, xscrollcommand=scrollx.set)
            self.txt.grid(row=0,column=0,sticky='nsew')
            scroll.config(command=self.txt.yview)
            scrollx.config(command=self.txt.xview)

            self.small_bp = tk.Frame(self.main_fr, relief='groove',borderwidth=5)
            self.small_bp.grid_columnconfigure(0, weight=1)
            self.small_bp.grid_columnconfigure(1, weight=3)
            self.small_bp.grid_columnconfigure(2, weight=3)
            self.small_bp.grid(row=0, column=0,sticky='nsew')

            labd = tk.Label(self.small_bp, text = "Duplicated rows:", anchor='e')
            labd.grid(row=0,column=0,sticky='nsew',pady=5)
            btkd = tk.Button(self.small_bp,text='Check', relief='groove', command=self.check_du)
            btkd.grid(row=0, column=1, sticky='nsew', padx=5,pady=5)
            btkd2 = tk.Button(self.small_bp,text='Remove', relief='groove',command=self.remove)
            btkd2.grid(row=0, column=2, sticky='nsew',padx=5,pady=5)

            labo = tk.Label(self.small_bp, text = "Remove outliers from", anchor='e')
            labo.grid(row=1, column=0, sticky='nsew')
            self.outl = tk.StringVar()
            self.outl.set("Column")
            dropmen7 = tk.OptionMenu(self.small_bp, self.outl, *self.controller.df.columns)
            dropmen7.config(relief='groove')
            dropmen7.grid(row=1, column=1, sticky='nsew')
            self.outl.trace("w", lambda x,y,z : self.on_select(mode=0))
            self.sym = tk.StringVar()
            dropmen6 = tk.OptionMenu(self.small_bp,self.sym, "<",">")
            dropmen6.config(relief='groove')
            dropmen6.grid(row=1, column=2, sticky='nsew')
            self.e1 = tk.Entry(self.small_bp, width=15)
            self.e1.grid(row=2, column=0,sticky='nse')
            self.e2 = tk.Entry(self.small_bp, width=10)
            self.e2.grid(row=2, column=2,sticky='nsew')
            b1 = tk.Button(self.small_bp, text="Replace with", relief='groove', command=self.repl)
            b1.grid(row=2, column=1, pady=3, sticky='nsew')
            l1= tk.Label(self.small_bp, text="Replacement value: (if you would like to set it to N/A,\n mean or median please put 'NA','mean','median' in entry box)")
            l1.grid(row=3, column=0, columnspan=3, sticky='nsew')

            self.framex = tk.Frame(self.main_fr, relief='groove', borderwidth=5)
            self.framex.grid(row=1, column=0,sticky='nsew')
            self.f, self.axes = plt.subplots(1, 2)
            self.canvas = FigureCanvasTkAgg(self.f, self.framex)
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self.canvas.draw()
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.framex)
            self.toolbar.update()
            self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            labn = tk.Label(self.small_bp, text='Filling NaN for column:', anchor='e')
            labn.grid(row=4,column=0, sticky='nsew')
            self.nana = tk.StringVar()
            self.nana.set("Column")
            self.nana.trace("w", lambda x,y,z : self.on_select(mode=1))
            dropmen7 = tk.OptionMenu(self.small_bp, self.nana, *self.controller.df.columns)
            dropmen7.config(relief='groove')
            dropmen7.grid(row=4, column=1, sticky='nsew')

            self.nejs = tk.StringVar()
            self.nejs.set("Method")
            self.nejs.trace('w',self.nanme)
            dropna = tk.OptionMenu(self.small_bp, self.nejs, "Drop", "Linear Interpolation", "Median", "Rolling Median"
                                          , "Rolling Mean","Specific Value")
            dropna.config(relief='groove')
            dropna.grid(row=4, column=2, sticky='nsew')
            self.na_frame = tk.Frame(self.small_bp)
            self.na_frame.grid_columnconfigure(0, weight=1)
            self.na_frame.grid_columnconfigure(1, weight=1)
            self.na_frame.grid_columnconfigure(2, weight=1)
            self.na_frame.grid(row=5, column=0 ,columnspan=3, sticky='nsew')
        else:
            self.controller.error()

    def check_du(self):
        self.txt.delete('1.0',tk.END)
        self.txt.insert(tk.END, self.controller.df[self.controller.df.duplicated() == True])

    def remove(self):
        self.controller.df.drop_duplicates(inplace=True)
        self.controller.df.reset_index(drop=True, inplace=True)
        print(self.controller.df.head())
        self.check_du()


    def on_select(self,mode=0,*args):

        self.axes[0].clear()
        self.axes[1].clear()


        self.txt.delete('1.0', tk.END)
        abc = ''
        if mode:
            self.x = self.nana.get()
            abc += 'Rows with NaN\n'
            abc += str(self.controller.df[self.x][self.controller.df[self.x].isna()])
        else:
            self.x = self.outl.get()
            abc += f'Statistic of {self.x}'
        self.txt.insert(tk.END, abc+'\n\n'+str(self.controller.df[self.x].describe())+'\n\n'+ str(self.controller.df[self.x]))
        if self.x != "":
            try:

                sns.distplot(pd.to_numeric(self.controller.df[self.x]), ax=self.axes[0])
                sns.boxplot(pd.to_numeric(self.controller.df[self.x]), orient='v', ax=self.axes[1])
                # sns.swarmplot(self.controller.df[self.x].dropna(), orient='v', color='black',ax=self.axes[1])
                plt.tight_layout()
            except (TypeError, ValueError):
                messagebox.showinfo("Error", "Cannot plot this data")
            self.axes[0].grid()
            self.axes[1].grid()
            self.canvas.draw()
            self.toolbar.update()

    def repl(self):

        value = float(self.e1.get())

        if self.e2.get().lower() == "na":
            if self.sym.get() == ">":
                self.controller.df[self.outl.get()][(self.controller.df[self.outl.get()] > value)] = np.NaN
            if self.sym.get() == "<":
                self.controller.df[self.outl.get()][(self.controller.df[self.outl.get()] < value)] = np.NaN

        elif self.e2.get().lower() == 'mean':
            if self.sym.get() == ">":
                self.controller.df[self.outl.get()][(self.controller.df[self.outl.get()] > value)] = self.controller.df[
                    self.outl.get()].mean()
            if self.sym.get() == "<":
                self.controller.df[self.outl.get()][(self.controller.df[self.outl.get()] < value)] = self.controller.df[
                    self.outl.get()].mean()
        elif self.e2.get().lower() == 'median':
            if self.sym.get() == ">":
                self.controller.df[self.outl.get()][(self.controller.df[self.outl.get()] > value)] = self.controller.df[
                    self.outl.get()].median()
            if self.sym.get() == "<":
                self.controller.df[self.outl.get()][(self.controller.df[self.outl.get()] < value)] = self.controller.df[
                    self.outl.get()].median()

        else:
            rep = float(self.e2.get())
            if self.sym.get() == ">":
                self.controller.df[self.outl.get()][(self.controller.df[self.outl.get()] > value)] = rep
            if self.sym.get() == "<":
                self.controller.df[self.outl.get()][(self.controller.df[self.outl.get()] < value)] = rep

        self.on_select(mode=0)

    def nanme(self, *args):
        for x in self.na_frame.winfo_children():
            x.destroy()

        btk = tk.Button(self.na_frame,text="Fill", command=self.fill, relief='groove')
        btk.grid(row=0, column=2, sticky='nsew', pady=5)

        if self.nejs.get() in ( "Rolling Median","Rolling Mean","Specific Value") :
            if self.nejs.get() != "Specific Value":
                l10 = tk.Label(self.na_frame, text="Window:", anchor='e')
                l10.grid(row=0, column=0, sticky='nse')
            else:
                l10 = tk.Label(self.na_frame, text="Value:", anchor='e')
                l10.grid(row=0, column=0, sticky='nse')

            self.e3 = tk.Entry(self.na_frame, width=5)
            self.e3.insert(1, 5)
            self.e3.grid(row=0, column=1, sticky='nsw')




    def fill(self):
        x = self.nana.get()
        if self.nejs.get() == 'Drop':
            self.controller.df.dropna(inplace=True)

        if self.nejs.get() == "Mean":
            self.controller.df[x].fillna(self.controller.df[x].mean(), inplace=True)

        if self.nejs.get() == "Median":
            self.controller.df[x].fillna(self.controller.df[x].median(), inplace=True)

        if self.nejs.get() == "Specific Value":
            if self.e3.get().isnumeric():
                self.controller.df[x].fillna(float(self.e3.get()), inplace=True)
            else:
                self.controller.df[x].fillna(self.e3.get(), inplace=True)

        if self.nejs.get() in ( "Rolling Median", "Rolling Mean"):
            win = int(self.e3.get())
            medDF = pd.DataFrame()

            if self.nejs.get() == "Rolling Median":
                medDF['med'] = self.controller.df[x].rolling(window=win, min_periods=1).median()
            else:
                medDF['med'] = self.controller.df[x].rolling(window=win, min_periods=1).mean()
            self.controller.df[x][(self.controller.df[x].isna())] = medDF['med']

        if self.nejs.get() == "Linear Interpolation":
            medDF = pd.DataFrame()
            medDF['med'] = self.controller.df[x].interpolate()
            self.controller.df[x][(self.controller.df[x].isna())] = medDF['med']

        self.on_select(mode=1)
