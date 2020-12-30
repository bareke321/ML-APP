import tkinter as tk
from data_load import Data_Load
from show_data import Show_Data
from stat_data import Stat_Data
from plot_data import Plot_Data
from create_par import Create_Par
from clean_tune import Clean_Tune
from train_test import Train_Test
from classifiers import Mlp, Knn, RandomForrest, SVM_SVC
from regressors import MLPreg, RandomReg, LinReg, SVM_SVR
from predict import Prediction
from grid_search import Grid_Search
import sys

class Button_Panel(tk.Frame):
    def __init__(self, parent, controller, *args, **kwargs):
        self.controller = controller
        super().__init__(parent, *args, **kwargs)
        self.keys = self.controller.frames_list
        print(self.keys)
        self.grid_columnconfigure(0, weight=1)
        btk = tk.Button(self, text="File", command=self.add_first, relief='groove',bg='white')
        btk.grid(row=0, column=0, sticky='nsew', pady=2)
        btk2 = tk.Button(self, text="Data preparation", relief='groove', command=self.add_second,bg='white')
        btk2.grid(row=7, column=0, sticky='nsew', pady=4)
        btk3 = tk.Button(self, text="Processing data", relief='groove', command=self.add_third,bg='white')
        btk3.grid(row=10, column=0, sticky='nsew', pady=4)
        btk4 = tk.Button(self, text="Classification", relief='groove', command=self.add_4,bg='white')
        btk4.grid(row=13, column=0, sticky='nsew', pady=4)
        btk5 = tk.Button(self, text="Regression", relief='groove', command=self.add_5,bg='white')
        btk5.grid(row=18, column=0, sticky='nsew', pady=4)
        btk6 = tk.Button(self, text="Predicton", relief='groove', command=self.add_6,bg='white')
        btk6.grid(row=25, column=0, sticky='nsew', pady=4)
        btk6 = tk.Button(self, text="Grid Search", relief='groove', command=lambda :self.controller.frames[Grid_Search].execute() if self.controller.full_pipe != [] else self.controller.error2(),bg='white')

        btk6.grid(row=23, column=0, sticky='nsew', pady=4)

        self.a = 1
        self.b = 1
        self.c =1
        self.d = 1
        self.e=1
        self.f=1


    def add_first(self):
        if self.a == 1:
            self.small_bp = tk.Frame(self, relief='groove', borderwidth=5, bg='white')
            self.small_bp.grid(row=1, column=0, rowspan=5, sticky='nsew')
            self.small_bp.grid_columnconfigure(0, weight=1)
            self.btk11 = tk.Button(self.small_bp, text="Load Data", relief='groove',command=self.controller.frames[Data_Load].file_name)
            self.btk11.grid(row=1, column=0, sticky='nsew')
            self.btk12 = tk.Button(self.small_bp, text="Show Data", relief='groove',command=self.controller.frames[Show_Data].show_df)
            self.btk12.grid(row=2, column=0, sticky='nsew')
            self.btk13 = tk.Button(self.small_bp, text="Clean Data", relief='groove', command=self.controller.clean_data)
            self.btk13.grid(row=3, column=0, sticky='nsew')
            self.btk14 = tk.Button(self.small_bp, text="Information about data", relief='groove',command=self.controller.frames[Stat_Data].stat)
            self.btk14.grid(row=4, column=0, sticky='nsew')
            self.btk15 = tk.Button(self.small_bp, text="Plot Data", relief='groove',command=self.controller.frames[Plot_Data].inter)
            self.btk15.grid(row=5, column=0, sticky='nsew')
            self.btk16 = tk.Button(self.small_bp, text="Exit", command=lambda: sys.exit(), relief='groove')
            self.btk16.grid(row=6, column=0, sticky='nsew')
            self.a = 0

        else:
            self.small_bp.destroy()
            self.a = 1




    def add_second(self):
        if self.b == 1:
            self.small_bp2 = tk.Frame(self, relief='groove', borderwidth=5, bg='white')
            self.small_bp2.grid(row=8, column=0, rowspan=2, sticky='nsew')
            self.small_bp2.grid_columnconfigure(0, weight=1)
            self.btk21 = tk.Button(self.small_bp2, text="Cleaning and tuning data",
                                   relief='groove', command=self.controller.frames[Clean_Tune].clean)
            self.btk21.grid(row=8, column=0, sticky='nsew')
            self.btk22 = tk.Button(self.small_bp2, text="Creating new parameter", relief='groove',
                                   command=self.controller.frames[Create_Par].create)
            self.btk22.grid(row=9, column=0, sticky='nsew')
            self.b = 0

        else:
            self.small_bp2.destroy()
            self.b = 1


    def add_third(self):
        if self.c == 1:
            self.small_bp3 = tk.Frame(self, relief='groove', borderwidth=5, bg='white')
            self.small_bp3.grid(row=11, column=0, rowspan=2, sticky='nsew')
            self.small_bp3.grid_columnconfigure(0, weight=1)
            self.btk31 = tk.Button(self.small_bp3, text="Split data for train, test set", relief='groove',
                                   command=self.controller.frames[Train_Test].show)
            self.btk31.grid(row=11, column=0, sticky='nsew')
            #self.btk32 = tk.Button(self.small_bp3, text="Statistic evaluation", relief='groove')
            #self.btk32.grid(row=12, column=0, sticky='nsew')
            self.c = 0
        else:
            self.small_bp3.destroy()
            self.c = 1


    def add_4(self):
        if self.d == 1:
            self.small_bp4 = tk.Frame(self, relief='groove', borderwidth=5, bg='white')
            self.small_bp4.grid(row=14, column=0, rowspan=2, sticky='nsew')
            self.small_bp4.grid_columnconfigure(0, weight=1)
            self.btk41 = tk.Button(self.small_bp4, text="MLP", relief='groove',
                                   command=lambda:self.controller.frames[Mlp].start() if self.controller.full_pipe != [] else self.controller.error2())
            self.btk41.grid(row=14, column=0, sticky='nsew')
            self.btk42 = tk.Button(self.small_bp4, text="Random Forest Classifier",
                                   relief='groove',
                                   command= lambda:self.controller.frames[RandomForrest].start() if self.controller.full_pipe != [] else self.controller.error2())
            self.btk42.grid(row=15, column=0, sticky='nsew')
            self.btk43 = tk.Button(self.small_bp4, text="K-nearest neighbour",
                                   relief='groove',
                                   command=lambda:self.controller.frames[Knn].start() if self.controller.full_pipe != [] else self.controller.error2())
            self.btk43.grid(row=16, column=0, sticky='nsew')
            self.btk44 = tk.Button(self.small_bp4, text="Support vector machine",
                                   relief='groove',
                                   command=lambda:self.controller.frames[SVM_SVC].start() if self.controller.full_pipe != [] else self.controller.error2())
            self.btk44.grid(row=17, column=0, sticky='nsew')
            self.d = 0
        else:
            self.small_bp4.destroy()
            self.d = 1

    def add_5(self):
        if self.e == 1:
            self.small_bp5 = tk.Frame(self, relief='groove', borderwidth=5, bg='white')
            self.small_bp5.grid(row=19, column=0, rowspan=2, sticky='nsew')
            self.small_bp5.grid_columnconfigure(0, weight=1)
            self.btk51 = tk.Button(self.small_bp5, text="MLP", relief='groove',
                                   command=lambda:self.controller.frames[MLPreg].start() if self.controller.full_pipe != [] else self.controller.error2())
            self.btk51.grid(row=19, column=0, sticky='nsew')
            self.btk52 = tk.Button(self.small_bp5, text="Random Forest Regressor", relief='groove',
                                   command=lambda:self.controller.frames[RandomReg].start() if self.controller.full_pipe != [] else self.controller.error2())
            self.btk52.grid(row=20, column=0, sticky='nsew')
            self.btk53 = tk.Button(self.small_bp5, text="Linear Regression", relief='groove',
                                   command=lambda:self.controller.frames[LinReg].start() if self.controller.full_pipe != [] else self.controller.error2())
            self.btk53.grid(row=21, column=0, sticky='nsew')
            self.btk54 = tk.Button(self.small_bp5, text="Support vector machine", relief='groove',
                                   command=lambda:self.controller.frames[SVM_SVR].start() if self.controller.full_pipe != [] else self.controller.error2())
            self.btk54.grid(row=22, column=0, sticky='nsew')
            self.e = 0
        else:
            self.small_bp5.destroy()
            self.e = 1

    def add_6(self):
        if self.f == 1:
            self.small_bp6 = tk.Frame(self, relief='groove', borderwidth=5, bg='white')
            self.small_bp6.grid(row=26, column=0, rowspan=2, sticky='nsew')
            self.small_bp6.grid_columnconfigure(0, weight=1)
            self.btk61 = tk.Button(self.small_bp6, text="Predict new data", relief='groove',
                                   command=lambda: self.controller.frames[Prediction].win_create() if self.controller.algho else self.controller.error3())
            self.btk61.grid(row=26, column=0, sticky='nsew')
            self.f = 0

        else:
            self.small_bp6.destroy()
            self.f = 1
