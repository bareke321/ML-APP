from tkinter import messagebox
from tkinter import filedialog
import tkinter as tk
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelBinarizer, StandardScaler, LabelEncoder,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from tkinter import messagebox

class DataFrameSelectorStat(BaseEstimator, TransformerMixin):
    def __init__(self,step=10):
        self.step = step
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        ab = 0
        a = 0

        flt_col = []
        for x in X.columns:
            if X[x].dtype != object:
                flt_col.append(x)
        X = X[flt_col]
        self.processing_columns = flt_col
        for x in range(self.step , X.shape[0], self.step ):
            wp = np.array([*X.iloc[a:x].mean().values, *X.iloc[a:x].median().values,*X.iloc[a:x].skew().values,
                          *X.iloc[a:x].kurtosis().values, *X.iloc[a:x].std().values, *X.iloc[a:x].var().values,
                          *X.iloc[a:x].max().values, *X.iloc[a:x].min().values,
                          *X.iloc[a:x].max().values - X.iloc[a:x].min().values,
                          *X.iloc[a:x].std().values/X.iloc[a:x].mean().values]).reshape(1,-1)
            if x==self.step:
                ab = pd.DataFrame(wp)
            else:
                ab = ab.append(pd.DataFrame(wp))
            a = x

        return ab.values


class Stat_Eval(tk.Frame):
    def __init__(self, parent, controller):
        self.controller = controller
        super().__init__(parent)
        self.keys = self.controller.frames_list

    def stra(self):
        for widget in self.winfo_children():
            widget.destroy()
        self.controller.show_frame(Stat_Eval)
        labt = tk.Label(self, text="Target column:", anchor='e')
        labt.grid(row=0, column=0, sticky='nsew', pady=5)
        self.target = tk.StringVar()
        drop_target = tk.OptionMenu(self, self.target, *self.controller.columns)
        drop_target.configure(relief='groove')
        drop_target.grid(row=0, column=1, sticky='nsew', pady=5)
        btk = tk.Button(self,text='go',command=self.cal)
        btk.grid(row=0, column=4, sticky='nsew', pady=5)
        lb = tk.Label(self, text="Step:")
        lb.grid(row=0,column=2)
        self.e1 = tk.Entry(self)
        self.e1.insert(1,10)
        self.e1.grid(row=0, column=3)
    def cal(self):
        self.controller.step = int(self.e1.get())
        self.controller.num_pipeline = [
            ('Select and Create', DataFrameSelectorStat(step=self.controller.step)),
            ("Imputer", SimpleImputer(strategy='mean')),
            ('Scaling', StandardScaler()),
            ]

        self.controller.fet_set = self.controller.df.drop(self.target.get(),axis=1)
        self.controller.processing_columns = self.controller.fet_set.columns
        self.controller.stan=1
        self.controller.eval_dt = 1
        self.controller.ml_type = 'Classification'
        self.controller.target_name = self.target.get()
        self.controller.target = self.controller.df[self.target.get()].copy()
        self.controller.full_pipe = Pipeline(self.controller.num_pipeline)
        self.controller.test_size = 0.2
        messagebox.showinfo("Info", "Done")




