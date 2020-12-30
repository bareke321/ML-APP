import tkinter as tk
import pandas as pd
import numpy as np
import tkinter as tk
import inspect
import webbrowser
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
import _thread
from sklearn.multiclass import OneVsRestClassifier
from classifiers import baseclassi
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelBinarizer, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, plot_confusion_matrix,classification_report, roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
#import sklearn
import multiprocessing
import sys
from sklearn.model_selection import train_test_split
from grid_search import Grid_Search
from button_panel import Button_Panel
from data_load import Data_Load
from start_page import Start_Page
from show_data import Show_Data
from stat_data import Stat_Data
from plot_data import Plot_Data
from create_par import Create_Par
from clean_tune import Clean_Tune
from train_test import Train_Test
from classifiers import Mlp, Knn, RandomForrest, SVM_SVC
from regressors import MLPreg, RandomReg, LinReg, SVM_SVR
from predict import Prediction
from tkinter import messagebox
from stat_eval import Stat_Eval
import threading
from joblib import Parallel, delayed

class Main_Program(tk.Tk):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        tk.Tk.wm_title(self, "ML")
        container = tk.Frame(self,relief='groove',borderwidth=5)
        container.pack(side="right", fill="both",expand = True)
        container.grid_columnconfigure(0, weight=1)
        container.grid_columnconfigure(1, weight=60)
        container.grid_rowconfigure(0, weight=1)
        #container.grid_rowconfigure(0, weight = 1)
        #container.grid_columnconfigure(0, weight=1)

        left_panel = tk.Frame(container, relief='groove', bg='white', borderwidth=5)
        left_panel.grid(column=0, row=0, sticky="nsew")
        left_panel.grid_columnconfigure(0, weight=1)
        #left_panel.grid_rowconfigure(0, weight = 1)


        right_panel = tk.Frame(container, relief='groove', bg='white', borderwidth=5)
        right_panel.grid(column=1, row=0, sticky="nsew")
        right_panel.grid_columnconfigure(0, weight=1)
        right_panel.grid_rowconfigure(0, weight = 1)

        #right_panel.grid_propagate(False)
        #right_panel.grid_rowconfigure(0, weight=1)

        self.frames = {}

        self.frames_list =[Data_Load, Start_Page, Show_Data, Stat_Data,Plot_Data,
                           Create_Par, Clean_Tune, Train_Test, Mlp, Knn, RandomForrest, SVM_SVC,
                           MLPreg, RandomReg, LinReg, SVM_SVR, Prediction,Stat_Eval, Grid_Search
                           ]

        for F in self.frames_list:
            frame = F(right_panel, self)
            self.frames[F] = frame

        self.show_frame(Start_Page)
        #self.frames[Start_Page].rotated()
        for F in self.frames:
            self.frames[F].grid(row=0, column =0, sticky ='nsew')


        btj = Button_Panel(left_panel,self)
        btj.pack(side="left", fill="both",expand = True)

        self.algho ={"-":""}
        self.full_pipe = []
        self.eval_dt =0
        self.encoder = 0

        menubar = tk.Menu(container)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Load Data", command = self.frames[Data_Load].file_name)
        filemenu.add_command(label="Show Data", command =self.frames[Show_Data].show_df)
        filemenu.add_command(label="Clean Data", command= self.clean_data)
        filemenu.add_command(label="Information about data",command=self.frames[Stat_Data].stat)
        filemenu.add_command(label="Plot Data",command=self.frames[Plot_Data].inter)
        #filemenu.add_command(label="Spin me", command=self.frames[Start_Page].rotated)
        filemenu.add_command(label="Exit", command=self.exit)
        menubar.add_cascade(label="File",menu= filemenu)
        preparation = tk.Menu(menubar, tearoff=0)

        preparation.add_command(label="Cleaning and tuning data", command=self.frames[Clean_Tune].clean)
        preparation.add_command(label="Creating new parameter",command=self.frames[Create_Par].create)

        menubar.add_cascade(label="Data preparation", menu=preparation)
        processing = tk.Menu(menubar, tearoff=0)
        processing.add_command(label="Split data fot train, test sets",command=self.frames[Train_Test].show)
        processing.add_command(label="Statistic evaluation",command=self.frames[Stat_Eval].stra)

        menubar.add_cascade(label="Processing data", menu=processing)
        ml = tk.Menu(menubar, tearoff=0)
        # ml.add_command(label = "Regression",command=lambda: print("XD"))
        # ml.add_command(label="Classification", command=lambda: print("XD"))

        menubar.add_cascade(label="Machine learning", menu=ml)
        classisub = tk.Menu(ml, tearoff=0)
        classisub.add_command(label="MLP",command= lambda: self.frames[Mlp].start() if self.full_pipe != [] else self.error2())
        classisub.add_command(label="Random Forest Classifier",command= lambda:self.frames[RandomForrest].start() if self.full_pipe != [] else self.error2())
        classisub.add_command(label="K-nearest neighbour", command=lambda:self.frames[Knn].start() if self.full_pipe != [] else self.error2())
        classisub.add_command(label="Support vector machine", command= lambda:self.frames[SVM_SVC].start() if self.full_pipe != [] else self.error2())
        ml.add_cascade(label="Classification", menu=classisub)
        regress = tk.Menu(ml, tearoff=0)
        regress.add_command(label="MLP", command= lambda:self.frames[MLPreg].start() if self.full_pipe != [] else self.error2())
        regress.add_command(label="Linear Regression", command= lambda:self.frames[LinReg].start() if self.full_pipe != [] else self.error2())
        regress.add_command(label="Random Forest Regresor", command= lambda:self.frames[RandomReg].start() if self.full_pipe != [] else self.error2())
        regress.add_command(label="Support vector machine", command= lambda:self.frames[SVM_SVR].start() if self.full_pipe != [] else self.error2())
        ml.add_cascade(label="Regression", menu=regress)
        ml.add_command(label="Grid Search", command= lambda: self.frames[Grid_Search].execute() if self.full_pipe != [] else self.error2())

        predict = tk.Menu(menubar, tearoff=0)
        predict.add_command(label="Predict new data", command=self.frames[Prediction].win_create)

        menubar.add_cascade(label="Predict", menu=predict)
        self.df = ''
        tk.Tk.config(self, menu=menubar)


    def show_frame(self,fr):
        frame = self.frames[fr]
        frame.tkraise()
    def exit(self):
        for thread in threading.enumerate():
            thread.join()
        sys.exit()
    def clean_data(self):
        #for widget in self.frames:
           # for x in self.frames[widget].winfo_children():
             #   x.destroy()
        self.show_frame(Start_Page)
        self.df =''
        self.algho = {}
        self.full_pipe = []
        self.stan = 1
    def error(self):
        messagebox.showinfo("Error", "Please load data first")
        self.show_frame(Start_Page)
    def error2(self):
        messagebox.showinfo("Error", "Please split data for test, train set first")
        self.show_frame(Start_Page)
    def error3(self):
        messagebox.showinfo("Error", "Please save ML algorithm first")
        self.show_frame(Start_Page)


if __name__ == '__main__':

    try:
        ab = Main_Program()
        ab.geometry("1280x820")
        ab.mainloop()
    except RuntimeError:
        pass
