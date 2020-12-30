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
from tkinter import filedialog
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
from multiprocessing import Queue, Process
from sklearn.metrics import make_scorer
import itertools
from sklearn.metrics import silhouette_score
import queue
from tkinter import messagebox
import threading
X = np.random.rand(10000,5)
#Y = np.linspace(1,10,10000)
Y = np.random.randint(1,10,10000)
from sklearn import linear_model, neural_network, neighbors, tree, svm, ensemble, cluster

#reg_list = ['RandomForestRegressor', 'LinearRegression', 'MLPRegressor']
reg_list = []
reg_dict = {}
#reg_dict = {'RandomForestRegressor':RandomForestRegressor, 'LinearRegression':LinearRegression, 'MLPRegressor':MLPRegressor}
site = "https://scikit-learn.org/stable/modules/generated/"
mod_dict = {"ensemble":ensemble,'linear_model':linear_model, "neural_network":neural_network, 'neighbors':neighbors,'tree':tree,"svm":svm,'cluster':cluster}
mod_l = ["ensemble",'linear_model', "neural_network", 'neighbors', 'tree', "svm"]
cluster_lt = []
for x in mod_dict:
    for y in [z for z in dir(mod_dict[x]) if z[0].isupper()]:
        reg_list.append(y)
        if x == 'cluster':
            cluster_lt.append(y)
        reg_dict[y] = eval("{}.{}".format(x,y))

cluster_lt.remove("FeatureAgglomeration")
cluster_lt.remove("SpectralBiclustering")
cluster_lt.remove("SpectralCoclustering")
for x in reg_dict:
    try:
        inspect.signature(reg_dict[x])
    except (TypeError ,ValueError) as e:
        reg_list.remove(x)
to_remove = ['BernoulliRBM','KNeighborsTransformer','LocalOutlierFactor','NearestNeighbors','NeighborhoodComponentsAnalysis',
             'RadiusNeighborsTransformer', 'BaseDecisionTree','VotingRegressor','VotingClassifier',
             'BaseEnsemble','IsolationForest','OneClassSVM','RandomTreesEmbedding','RidgeCV','RidgeClassifierCV',
             'StackingClassifier','StackingRegressor','RadiusNeighborsRegressor']
class_list = []
regres_list = []

for x in to_remove:
    reg_list.remove(x)

reg_list.sort()
for x in reg_list:
    if "Classifier" in x or "Logistic" in x or "SVC" in x:
        class_list.append(x)
    if ("Regressor" in x or "Regression" in x or "SVR" in x) and ("Logistic" not in x):
        regres_list.append(x)



def fetch_par(func="", rec = False):
    if rec:
        par_list = rec
    else:
        par_list = str(inspect.signature(func)).replace('(',"").replace(')',"").split(', ')
        par_list.remove('*')


    try:
        par_list.remove('**kwargs')
    except ValueError:
        pass
    par_dict = {}
    for x in par_list:
        #print(x)
        val = x.split('=')[1]
        try:
            val = float(val)
            try:
                if val == int(val):
                    val = int(val)
            except OverflowError:
                pass
        except ValueError:
            if val[0] =="'":
                val = val[1:-1]

        par_dict[x.split('=')[0].strip()] = val

    return par_dict

def conv(val):
    val = str(val).strip()
    if val.split('(')[0] in reg_list:
        #tutaj pomyslec trza
        if "(" in val and len(val.split('(')[1]) > 3:
            par = val.split('(')[1][:-1].split(',')
            par_dict = fetch_par(rec=par)
            #print(par_dict)
            k_dt = {}
            #print(type(par_dict))
            for x in par_dict:
                k_dt[x] = conv(par_dict[x])
            return reg_dict[val.split('(')[0]](**par_dict)
        else:

            return reg_dict[val.split('(')[0]]()
    if ',' in val:
        return [int(x) for x in val.split(',') if x != '']
    try:
        val = float(val)
        try:
            if val == int(val):
                val = int(val)
        except ValueError:
            pass
    except ValueError:
        if val.lower() == "true":
            val = True
        elif val.lower() == "false":
            val = False
        elif val.lower() == "none":
            val = None

    return val


class dropmenu(tk.OptionMenu):
    def __init__(self, parent, tab_list):
        self.var = tk.StringVar()
        #self.var.set(tab_list[0])
        super().__init__(parent, self.var, *tab_list)
        self.configure(relief='groove')


class Grid_Search(baseclassi):
    def __init__(self, parent=None, controller=None ):
        super().__init__(parent,controller = controller)
        self.controller = controller
        #.keys = self.controller.frames_list
        self.queue = Queue()
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=300)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

    def main_clf(self):
        super().main_clf()
        self.b12.destroy()
        self.b14.destroy()
        self.b15.destroy()

    def execute(self):
        self.controller.show_frame(Grid_Search)
        self.main_clf()
        tk.Label(self.main_fr, text="Please choose type of algorithm").grid(row=0,column=0,sticky='nsew',columnspan=2)
        self.reg_class = dropmenu(self.main_fr,["Regressor","Classifier","Clustering","All"])
        self.reg_class.grid(row=1,column=0,sticky='nsew',columnspan=2)
        self.reg_class.var.trace('w',self.app_dis)
        self.ab = 0
    def app_dis(self,*args):
        self.controller.stan = 1
        if self.ab:
            self.ab.destroy()
        if self.reg_class.var.get() == "Regressor":
            self.ab = dropmenu(self.main_fr,regres_list)
        if self.reg_class.var.get() == "Classifier":
            self.ab = dropmenu(self.main_fr,class_list)
        if self.reg_class.var.get() == "All":
            self.ab = dropmenu(self.main_fr,reg_list)
        if self.reg_class.var.get() == "Clustering":
            self.ab = dropmenu(self.main_fr, cluster_lt)
        self.ab.var.trace('w',self.regre)
        self.ab.grid(row=2,column=0,sticky='nsew',columnspan=2)
        self.side_fr = tk.Frame(self.main_fr,relief='groove',borderwidth=5)
        self.side_fr.grid(row=4,column=0,columnspan=2,rowspan=10,sticky='nsew')

    def regre(self, *args):
        for x in self.side_fr.winfo_children():
            x.destroy()
        self.Entry_create(reg_dict[self.ab.var.get()])

    def Entry_create(self, func):
            self.func = func
            self.Entries = {}
            i = 0
            j = 0
            self.par_dict = fetch_par(func)
            for x in self.par_dict:
                tk.Label(self.side_fr,text = "{}: ".format(x)).grid(row=i,column=j)
                j += 1
                self.Entries[x] = tk.Entry(self.side_fr, width=60)
                self.Entries[x].grid(row=i,column=j)
                self.Entries[x].insert(0,str(self.par_dict[x]))
                j = 0
                i += 1
            tk.Label(self.main_fr, text='Use ";" as separator between parameters').grid(row=3, column=0, sticky='nsew',
                                                                                columnspan=2)

            self.str_grd = tk.Button(self.side_fr, relief='groove', text="Start grid search",command= lambda : threading.Thread(target=(self.crosfit),args=(),daemon=True).start())
            #self.str_grd = tk.Button(self.side_fr, relief='groove',text="Start grid search", command=lambda: _thread.start_new_thread(self.crosfit,()))
            self.str_grd.grid(row=i, column=j+1, sticky='nsew')
            tk.Button(self.side_fr,text='Help',relief='groove',command=lambda: webbrowser.open("".join([site,".".join([func.__module__.split('._')[0],func.__name__,"html"])]))).grid(row=i,column=j,sticky='nsew')
            self.plot_sum = tk.Button(self.side_fr, relief='groove',text="Plot summarize", command= self.plot_summary)
            self.plot_sum['state'] = 'disable'
            self.plot_sum.grid(row=i+1, column=j,columnspan=2, sticky='nsew')
            if (self.reg_class.var.get() == "Clustering") | (self.controller.ml_type == "Clustering"):
                self.b16.destroy()
                self.e15.destroy()
                b15 = tk.Button(self.main_fr, text="Save result",relief='groove',command=self.ch_model).grid(row=16,column=0,columnspan=2,sticky='nsew')
                self.plot_sum.destroy()

    def crosfit(self):
        self.str_grd['state'] = 'disable'
        self.get_data(encoder=LabelEncoder)
        self.params = {}
        for x in self.Entries:
            if self.Entries[x].get() != str(self.par_dict[x]):
                self.params[x] = list(map(conv, self.Entries[x].get().split(';')))
        self.param_grid = [self.params]

        self.alg = self.func()
        self.grid_search = GridSearchCV(self.alg, self.param_grid, cv=5, n_jobs=1)
        self.fit_clf()

    def fit_clf(self):
        try:
            if self.controller.ml_type != "Clustering":
                self.grid_search.fit(self.controller.X_train, self.controller.y_train)
                self.clf = self.func(**self.grid_search.best_params_)
                self.clf.fit(self.controller.X_train, self.controller.y_train)
                self.scoring()
            else:
                self.cluss_gridig()
                self.scoring()
            self.str_grd['state'] = 'normal'
        except Exception as p:
            messagebox.showinfo("Error", f"{p}")
            self.str_grd['state'] = 'normal'

    def cluss_gridig(self):
        self.dict_list = []
        i = int(np.product(list(map(len, self.params.values()))))
        for x in range(int(np.product(list(map(len, self.params.values()))))):
            self.dict_list.append({})
        arg_name = list(self.params.keys())
        iter_trt = 0
        for per in (list(itertools.product(*list(self.params.values())))):
            for i in range(len(self.params)):
                self.dict_list[iter_trt][arg_name[i]] = per[i]
            iter_trt += 1
        self.score_tab = []
        kwk = 0
        self.clf_list = []
        for arg_dict in self.dict_list:
            clf = self.func(**arg_dict)
            clf.fit(self.controller.X_cl)
            self.clf_list.append(clf)
            self.score_tab.append([silhouette_score(self.controller.X_cl,clf.labels_),kwk])
            kwk += 1
        self.score_tab.sort(reverse=True)
        print(self.score_tab,self.dict_list[self.score_tab[0][1]])
    def ch_model(self):
        self.bd = tk.Toplevel()
        self.bd.geometry("300x200")
        self.bd.title("Choose Model")
        self.dp_mod = dropmenu(self.bd,list(range(len(self.score_tab))))
        self.dp_mod.pack(side='top',expand=True,fill='both')
        tk.Button(self.bd,text="OK",command=self.giu_change).pack(side='top',expand=True,fill='both')
    def giu_change(self):
        df_to_drop = self.controller.fet_set
        df_to_drop['Labels'] = self.clf_list[int(self.dp_mod.var.get())].labels_
        self.bd.destroy()
        filename = filedialog.asksaveasfilename(initialdir = "/",title ="abc",filetypes = (("csv files","*.csv"),("all files","*.*")))
        df_to_drop.to_csv(".".join([filename,'csv']), index=False)
        messagebox.showinfo("Success", f"File has been saved on {filename}")
    def plot_summary(self):
        self.canvas.draw()
    def scoring(self):
        if self.controller.ml_type != "Clustering":
            scr = 'Best Paramters: \n' + str(self.grid_search.best_params_) + '\n\n'
            self.f, self.axes = plt.subplots(1, 1)
            self.canvas = FigureCanvasTkAgg(self.f, self.sec_fr)
            self.canvas.get_tk_widget().grid(row=1, column=0, columnspan=2, sticky='nsew')
            self.axes.clear()
        else:
            scr = 'Best Paramters: \n' + str(self.dict_list[self.score_tab[0][1]]) + '\n\n'
            scr += "Index       Silhouette Score            Parameters\n"
            scr += "\n".join([str(y)+'             ' +str(x)[:5]+'                       '+str(self.dict_list[y]) for x, y in self.score_tab])

        if self.controller.ml_type == "Classification":
            scr += "Score:\n" + str(
                self.clf.score(self.controller.X_test, self.controller.y_test)) + "\n\nConfusion Matrix\n"
            y_pred = self.controller.encoder.inverse_transform(self.clf.predict(self.controller.X_test))
            y_true = self.controller.encoder.inverse_transform(self.controller.y_test)
            self.conf = confusion_matrix(y_true, y_pred, labels=self.controller.encoder.classes_)
            class_report = classification_report(y_true, y_pred,
                                                 target_names=[str(x) for x in self.controller.encoder.classes_])
            scr += str(pd.DataFrame(self.conf, columns=self.controller.encoder.classes_,
                                    index=self.controller.encoder.classes_))
            scr += '\n\n' + str(class_report)
            self.lab = list(self.controller.encoder.classes_)

            sns.heatmap(self.conf, square=True, annot=True, cbar=False,ax=self.axes,fmt='d')
            self.axes.set_xticklabels(labels=self.lab)
            self.axes.set_yticklabels(labels=self.lab)
            self.axes.set_ylabel("Actual class")
            self.axes.set_xlabel("Predicted class")
            self.plot_sum['state'] = 'normal'



        if self.controller.ml_type == 'Regression':
            self.clf.fit(self.controller.X_train, self.controller.y_train)
            self.y_pred = self.clf.predict(self.controller.X_test)
            self.y_true = self.controller.y_test
            scr += " Root Mean Square Error:\n" + str(
                mean_squared_error(self.y_true, self.y_pred) ** 0.5) + u"\n\nR\u00b2: " + str(
                r2_score(self.y_true, self.y_pred)) + "\n\n"
            scr += "Explained variance: " + str(explained_variance_score(self.y_true, self.y_pred))

            x = range(len(self.y_pred))
            if len(self.y_pred) > 1000:
                s = 5
            else:
                s = 12
            self.axes.scatter(self.y_true, self.y_pred, label="Predicted", s=s, color='red', marker='*')
            self.axes.legend(loc='best')
            self.axes.set_ylabel("Predicted")
            self.axes.set_xlabel("Actual")
            self.axes.grid()
            self.plot_sum['state'] = 'normal'
        self.txt.delete('1.0', tk.END)
        self.txt.insert('1.0', scr)
        if self.controller.ml_type != "Clustering":
            self.txt.tag_add('center', "1.0", tk.END)
        plt.tight_layout()



