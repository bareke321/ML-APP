import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tkinter as tk
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder,OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
import sys
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)

class dropmenu(tk.OptionMenu):
    def __init__(self, parent, tab_list):
        self.var = tk.StringVar()
        #self.var.set(tab_list[0])
        super().__init__(parent, self.var, *tab_list)
        self.configure(relief='groove')


class DataFrameSelectorstr(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        ab = []
        for x in X.columns:
            if X[x].dtype == object:
                X[x] = LabelBinarizer().fit_transform(X[x])
                ab.append(x)
        return X[ab].values


class DataFrameSelectorflt(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        ab = []
        for x in X.columns:
            if X[x].dtype != object:
                ab.append(x)
        return X[ab].values

class MylabelBinarizer(TransformerMixin):
    def __init__(self, *args,**kwargs):
        self.encoder = LabelEncoder(*args,**kwargs)
    def fit(self,x,y=0):
        self.encoder.fit(x)
        return self
    def transform(self,x,y=0):
        return self.encoder.transform(x)

class Train_Test(tk.Frame):
    def __init__(self, parent, controller ):
        super().__init__(parent,  relief='groove', borderwidth=5, bg='white')
        self.controller = controller
        self.keys = self.controller.frames_list
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

    def show(self):
        self.controller.show_frame(Train_Test)
        for widget in self.winfo_children():
            widget.destroy()
        if type(self.controller.df) != str:

            self.main_fr = tk.Frame(self, relief='groove', borderwidth=5)
            self.main_fr.grid(row=0, column=0, sticky='new')
            self.main_fr.grid_columnconfigure(0, weight=1)
            self.main_fr.grid_columnconfigure(1, weight=5)
            self.main_fr.grid_columnconfigure(2, weight=1)
            self.main_fr.grid_columnconfigure(3, weight=5)
            self.main_fr.grid_columnconfigure(4, weight=1)


            self.sec_fr = tk.Frame(self, relief='groove', borderwidth=5, bg='white')
            self.sec_fr.grid(row=0, column=1, sticky='nsew')
            self.sec_fr.grid_rowconfigure(0, weight=1)
            self.sec_fr.grid_columnconfigure(0, weight=1)

            scroll = tk.Scrollbar(self)
            scrollx = tk.Scrollbar(self, orient=tk.HORIZONTAL)
            scroll.grid(row=0, column=125, rowspan=50, sticky='ns')
            scrollx.grid(row=15, column=1, columnspan=160, sticky='ew')

            self.txt = tk.Text(self.sec_fr, borderwidth=0,wrap=tk.NONE, yscrollcommand=scroll.set, xscrollcommand=scrollx.set)
            self.txt.grid(row=0,column=0,sticky='nsew')
            scroll.config(command=self.txt.yview)
            scrollx.config(command=self.txt.xview)

            l1 = tk.Label(self.main_fr,text="Dataset columns",anchor='w')
            l1.grid(row=0,column=1,sticky='w')
            l2 = tk.Label(self.main_fr, text="Features columns",anchor='w')
            l2.grid(row=0,column=3,sticky='w')

            scrollcol = tk.Scrollbar(self.main_fr)
            scrollcol.grid(row=1,column=0, rowspan=2,sticky='ns')

            scrollfet = tk.Scrollbar(self.main_fr)
            scrollfet.grid(row=1,column=5, rowspan=2,sticky='ns')

            self.col = tk.Listbox(self.main_fr,yscrollcommand=scrollcol.set,selectmode=tk.EXTENDED,
                                  relief='groove', borderwidth=5)
            self.col.insert(tk.END, *self.controller.df.columns)
            self.col.grid(row=1,column=1,rowspan=2, sticky='nsew')
            self.fet = tk.Listbox(self.main_fr,yscrollcommand=scrollfet.set,selectmode=tk.EXTENDED,
                                  relief='groove', borderwidth=5)
            self.fet.grid(row=1,column=3,rowspan=2, sticky='nsew')
            scrollcol.config(command=self.col.yview)
            scrollfet.config(command=self.fet.yview)

            btk1 = tk.Button(self.main_fr, text=">",relief='groove',command=self.move_to_fet)
            btk1.grid(row=1,column=2,sticky='sew')
            btk2 = tk.Button(self.main_fr, text="<",relief='groove', command=self.move_back)
            btk2.grid(row=2,column=2,sticky='new')

            fram_type = tk.Frame(self.main_fr)
            fram_type.grid_columnconfigure(0, weight=1)
            fram_type.grid_columnconfigure(1, weight=1)
            fram_type.grid(row=3, column=0, columnspan=6, sticky='nsew')
            tk.Label(fram_type,text = "Type of machine learning", anchor='e').grid(row=0, column=0, sticky='nsew',pady=5)
            self.type_drop = dropmenu(fram_type,["Regression", "Classification","Clustering"])
            self.type_drop.grid(row=0, column=1, sticky='nsew',pady=5)
            self.type_drop.var.trace('w',self.targ)


            self.fram = tk.Frame(self.main_fr)
            self.fram.grid_columnconfigure(0, weight=1)
            self.fram.grid_columnconfigure(1, weight=1)
            self.fram.grid(row=4,column=0,columnspan=6,sticky='nsew')

            labt = tk.Label(self.fram, text="Target column:",anchor='e')
            labt.grid(row=0, column=0, sticky='nsew',pady=5)
            self.target = tk.StringVar()
            self.drop_target = tk.OptionMenu(self.fram, self.target, *self.col.get(0, tk.END))
            self.drop_target.configure(relief='groove')
            self.drop_target.grid(row=0, column=1, sticky='nsew', pady=5)

            self.nanopt = tk.StringVar()
            self.labn = tk.Label(self.fram,text="Please choose method to fill NaN", anchor='e')
            self.nandrop = tk.OptionMenu(self.fram, self.nanopt, "Mean","Median","Most Frequent",
                                         "Constant", "Drop")
            self.nandrop.configure(relief='groove')
            self.nanopt.trace('w',self.on_change_na)
            self.franan = tk.Frame(self.fram)
            self.franan.grid(row=2,column=0,columnspan=2,sticky='nsew')
            abc = self.controller.df.isna().any(axis=0)
            abc = abc[abc == True].index
            print(abc)
            if len(abc):
                self.labn.grid(row=1, column=0,sticky='nsew')
                self.nandrop.grid(row=1, column=1,sticky='nsew')

            labe = tk.Label(self.fram,text="Please provide percent of test set:", anchor='e')
            labe.grid(row=3,column=0,sticky='nsew',pady=5)
            self.test_size = tk.Entry(self.fram, width=10)
            self.test_size.insert(1,20)
            self.test_size.grid(row=3, column=1, sticky='nsw',pady=5)
            btk3 = tk.Button(self.fram, text='Apply',relief='groove', command=self.create_pipe)
            btk3.grid(row=10,column=1,sticky='nsew',pady=5)

            labsca = tk.Label(self.fram, text="Standard Scaler:", anchor='e')
            labsca.grid(row=5,column=0,sticky='nsew',pady=5)
            self.sca = tk.StringVar()
            self.sca.set("No")
            dropsca = tk.OptionMenu(self.fram, self.sca,"Yes","No")
            dropsca.configure(relief='groove')
            dropsca.grid(row=5,column=1,sticky='nsw',pady=5)

            labca = tk.Label(self.fram, text="PCA:", anchor='e')
            labca.grid(row=7,column=0,sticky='nsew',pady=5)
            self.pca = tk.StringVar()
            self.pca.set("No")
            droppca = tk.OptionMenu(self.fram, self.pca,"Yes","No")
            droppca.configure(relief='groove')
            droppca.grid(row=7,column=1,sticky='nsw',pady=5)
            self.pca.trace('w', self.on_change)
            self.framka = tk.Frame(self.fram)
            self.framka.grid(row=8,column=0,columnspan=2,sticky='nsew')
        else:
            self.controller.error()

    def targ(self,*args):

        if self.type_drop.var.get() == 'Clustering':
            self.target.set("")
            self.drop_target.configure(state='disabled',bg='black')
        else:
            self.drop_target.configure(state='normal',bg='#F0F0F0')
    def move_to_fet(self):
        print(self.col.curselection())
        if self.col.curselection():
            self.txt.delete('1.0',tk.END)
            #self.fet.insert(tk.END,*self.col.get(self.col.curselection()[0],self.col.curselection()[-1]))
            values = [self.col.get(x) for x in self.col.curselection()]
            self.fet.insert(tk.END,*values)
            #self.col.delete(self.col.curselection()[0], self.col.curselection()[-1])
            self.col.delete(0,tk.END)
            columns = list(self.controller.df.columns)
            for x in self.fet.get(0,tk.END):
                columns.remove(x)
            self.col.insert(0,*columns)

            self.txt.insert(tk.END, self.controller.df[list(self.fet.get(0, tk.END))])
            self.drop_target.destroy()
            self.target.set("")
            self.drop_target = tk.OptionMenu(self.fram, self.target, *self.col.get(0, tk.END))
            self.drop_target.configure(relief='groove')
            self.drop_target.grid(row=0, column=1, sticky='nsew')

    def move_back(self):
        if self.fet.curselection():
            self.txt.delete('1.0', tk.END)
            values = [self.fet.get(x) for x in self.fet.curselection()]
            self.col.insert(tk.END,*values)
            columns = list(self.fet.get(0,tk.END))
            self.fet.delete(0, tk.END)
            for x in values:
                columns.remove(x)
            self.fet.insert(0,*columns)

            self.txt.insert(tk.END, self.controller.df[list(self.fet.get(0, tk.END))])
            self.drop_target.destroy()
            self.target.set("")
            self.drop_target = tk.OptionMenu(self.fram, self.target, *self.col.get(0, tk.END))
            self.drop_target.configure(relief='groove')
            self.drop_target.grid(row=0, column=1, sticky='nsew')
    def on_change(self,*args):
        for x in self.framka.winfo_children():
            x.destroy()
        if self.pca.get() == "Yes":
            n_co = tk.Label(self.framka,text="n_components (use 0-1.0 to get expected variation):")
            n_co.grid(row=0,column=0,sticky='nsew')
            self.pc_en = tk.Entry(self.framka,width=10)
            self.pc_en.insert(1,0.95)
            self.pc_en.grid(row=0,column=1,sticky='nsw')

    def on_change_na(self, *args):
        for x in self.franan.winfo_children():
            x.destroy()
        if self.nanopt.get() == "Constant":
            labnann = tk.Label(self.franan, text="Please indicate Constant Value:", anchor='e')
            labnann.grid(row=0,column=0,sticky='nsew')
            self.con_nan = tk.Entry(self.franan,width=10)
            self.con_nan.insert(1,0)
            self.con_nan.grid(row=0,column=1,sticky='nsw')

    def create_pipe(self):

        self.controller.num_pipeline = [
            ('Select', DataFrameSelectorflt()),]
        self.controller.test_size = float(self.test_size.get())/100
        self.controller.stan=1
        self.controller.stan_reg = 1
        self.controller.eval_dt = 0
        self.controller.processing_columns = list(self.fet.get(0, tk.END))
        self.controller.ml_type = self.type_drop.var.get()
        col = self.controller.processing_columns[:]
        if self.type_drop.var.get() != "Clustering":
            col.append(self.target.get())
        df = self.controller.df[col]
        print(self.controller.test_size)

        if self.nanopt.get():
            if self.nanopt.get() in ("Mean", "Median", "Most Frequent", "Constant"):
                start = self.nanopt.get().lower()
                if self.nanopt.get() == "Constant":
                    self.controller.num_pipeline.append(("Imputer", SimpleImputer(strategy=start, fill_value=float(self.con_nan.get()))))
                elif self.nanopt.get() == "Most Frequent":
                    self.controller.num_pipeline.append(("Imputer", SimpleImputer(strategy="most_frequent")))
                else:
                    self.controller.num_pipeline.append(("Imputer", SimpleImputer(strategy=start)))

            elif self.nanopt.get() == 'Drop':
                df.dropna(inplace=True)
                start = "mean"
        else:
            start = "most_frequent"
            self.controller.num_pipeline.append(("Imputer", SimpleImputer(strategy=start)))
            
        self.controller.fet_set = df[list(self.fet.get(0, tk.END))].copy()
        if self.type_drop.var.get() != "Clustering":
            self.controller.target_name = self.target.get()
            self.controller.target = df[self.target.get()].copy()



        for x in self.controller.fet_set.columns:
            if self.controller.fet_set[x].dtype == object:
                self.controller.cat_pipeline = Pipeline([
                    ('Select', DataFrameSelectorstr()),
                    ("Imputer", SimpleImputer(strategy='most_frequent'))

                ])
                break
            self.controller.cat_pipeline = ''
        self.controller.num_pipeline = Pipeline(self.controller.num_pipeline)

        if self.controller.cat_pipeline:
            self.controller.full_pipe = FeatureUnion(transformer_list=[
                ("num_pipeline", self.controller.num_pipeline),
                ("cat_pipeline", self.controller.cat_pipeline)
            ])
        else:
            self.controller.full_pipe = self.controller.num_pipeline

        if self.sca.get() == "Yes":
            self.controller.full_pipe = Pipeline([("Feature Selection",self.controller.full_pipe),('Scaling', StandardScaler())])
        if self.pca.get() == "Yes":
            if float(self.pc_en.get()) < 1:
                self.controller.full_pipe = Pipeline([("Feature Selection",self.controller.full_pipe),("PCA",PCA(n_components=float(self.pc_en.get())))])
            else:
                self.controller.full_pipe =Pipeline([("Feature Selection",self.controller.full_pipe),("PCA", PCA(n_components=int(self.pc_en.get())))])


        self.txt.delete('1.0', tk.END)
        self.txt.insert(tk.END, "Full pipeline:\n" + str(self.controller.full_pipe))










