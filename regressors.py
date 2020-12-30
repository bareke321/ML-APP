from sklearn.preprocessing import StandardScaler, LabelBinarizer, OneHotEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from tkinter import messagebox
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, plot_confusion_matrix,classification_report, roc_curve
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd
import numpy as np
import tkinter as tk
from classifiers import baseclassi
def PolynomialRegression(degree, **kwargs):
    return make_pipeline(PolynomialFeatures(degree=degree),
                LinearRegression(**kwargs))


Big_Font = ("Times New Roman", 18)
med_font = ("Times New Roman",13)

actMLP = ['relu','identity', 'tanh', 'logistic']
solvMLP = ['adam','lbfgs', 'sgd']
learnrt = ['constant','invscaling','adaptive']
scorclas = ['accuracy','balanced_accuracy','average_precision','neg_brier_score']
crit = ['gini', 'entropy']
crit_reg = ['mse', 'mae']
kernels = ['rbf', 'linear', 'poly', 'sigmoid']
gammas = ['scale','auto']

class Base_Regress(baseclassi):
    def __init__(self, parent, controller):
        baseclassi.__init__(self, parent, controller)

    def main_clf(self):
        baseclassi.main_clf(self)
        self.f, self.axes = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.f, self.sec_fr)
        self.canvas.get_tk_widget().grid(row=1, column=0, columnspan=2, sticky='nsew')
        self.canvas.draw()
    def testcheck(self):
        self.b14['state'] = 'disable'
        self.clf.fit(self.controller.X_train, self.controller.y_train)
        self.y_pred = self.clf.predict(self.controller.X_test)
        self.y_true = self.controller.y_test
        scr = " Root Mean Square Error:\n" + str(
            mean_squared_error(self.y_true, self.y_pred)**0.5) + u"\n\nR\u00b2: "+str(r2_score(self.y_true, self.y_pred))+"\n\n"
        scr += "Explained variance: "+ str(explained_variance_score(self.y_true,self.y_pred))

        self.txt.delete('1.0', tk.END)
        self.txt.insert('1.0',scr)
        self.txt.tag_add('center', "1.0", tk.END)
        self.b14['state'] = 'normal'
        self.b15['state'] = 'normal'
    def summr(self):
        x = range(len(self.y_pred))
        if len(self.y_pred) > 1000:
            s = 5
        else:
            s = 12
        self.axes.clear()
        self.axes.scatter(self.y_true, self.y_pred, label= "Predicted", s=s,color='red',marker='*')
        #self.axes.scatter(x, self.y_true, label="Actual",s=s, alpha=0.7)
        #self.axes.xaxis.set_ticklabels([])
        #self.axes.xaxis.set_ticks([])
        self.axes.legend(loc='best')
        self.axes.set_ylabel("Predicted")
        self.axes.set_xlabel("Actual")
        self.axes.grid()
        self.f.tight_layout()
        self.canvas.draw()

    def score_val(self):

        self.b12['state'] = 'disable'
        self.get_data()
        scoval = cross_val_score(self.clf,self.controller.X_train, self.controller.y_train, cv=3,
                                     n_jobs =self.n_jobs, scoring = 'neg_mean_squared_error')

        self.txt.delete('1.0',tk.END)
        self.txt.insert('1.0',"Score from cross validation:"+str(np.sqrt(-scoval)))
        self.b12['state'] = 'normal'
        self.b14['state'] = 'normal'
"""
    def get_data(self):
        if self.controller.stan_reg:
            self.controller.fet_af = self.controller.full_pipe.fit_transform(self.controller.fet_set)
            self.controller.X_train, self.controller.X_test, self.controller.y_train, self.controller.y_test = \
                train_test_split(self.controller.fet_af, self.controller.target, test_size=self.controller.test_size)
            self.controller.stan_reg = 0
"""

class MLPreg(Base_Regress):
    def start(self):
        self.main_clf()
        self.controller.show_frame(MLPreg)
        l = tk.Label(self.main_fr, text='Parameters for MLP:', font=Big_Font)
        l.grid(row=0, column=0,columnspan=2, sticky='nsew', pady=8)

        l1 = tk.Label(self.main_fr, text="Hidden layer sizes:")
        l1.grid(row=1, column=0, sticky='e')
        self.e1 = tk.Entry(self.main_fr, width=20)
        self.e1.insert(1, ('100,'))
        self.e1.grid(row=1, column=1, sticky='w')

        l2 = tk.Label(self.main_fr, text="Activation function:")
        l2.grid(row=2, column=0, sticky='e')
        self.Activation = tk.StringVar()
        self.Activation.set(actMLP[0])
        dropmen2 = tk.OptionMenu(self.main_fr, self.Activation, *actMLP)
        dropmen2.configure(relief='groove')
        dropmen2.grid(row=2, column=1, sticky='nsew')

        l3 = tk.Label(self.main_fr, text="Solver for weight optimization:")
        l3.grid(row=3, column=0, sticky='e')
        self.solver = tk.StringVar()
        self.solver.set(solvMLP[0])
        dropmen3 = tk.OptionMenu(self.main_fr, self.solver, *solvMLP)
        dropmen3.configure(relief='groove')
        dropmen3.grid(row=3, column=1, sticky='nsew')

        l4 = tk.Label(self.main_fr, text="Alpha:")
        l4.grid(row=4, column=0, sticky='e')
        self.e4 = tk.Entry(self.main_fr, width=10)
        self.e4.insert(1, 0.0001)
        self.e4.grid(row=4, column=1, sticky='w')

        l5 = tk.Label(self.main_fr, text="Batch size:")
        l5.grid(row=5, column=0, sticky='e')
        self.e5 = tk.Entry(self.main_fr, width=10)
        self.e5.insert(1, 'auto')
        self.e5.grid(row=5, column=1, sticky='w')

        l6 = tk.Label(self.main_fr, text="Learning rate:")
        l6.grid(row=6, column=0, sticky='e')
        self.learning_rate = tk.StringVar()
        self.learning_rate.set(learnrt[0])
        dropmen6 = tk.OptionMenu(self.main_fr, self.learning_rate, *learnrt)
        dropmen6.configure(relief='groove')
        dropmen6.grid(row=6, column=1, sticky='nsew')

        l7 = tk.Label(self.main_fr, text="Initial learning rate:")
        l7.grid(row=7, column=0, sticky='e')
        self.e7 = tk.Entry(self.main_fr, width=10)
        self.e7.insert(1, 0.001)
        self.e7.grid(row=7, column=1, sticky='w')

        l8 = tk.Label(self.main_fr, text="Number of maximum iterations:")
        l8.grid(row=8, column=0, sticky='e')
        self.e8 = tk.Entry(self.main_fr, width=10)
        self.e8.insert(1, 200)
        self.e8.grid(row=8, column=1, sticky='w')

        l9 = tk.Label(self.main_fr, text="Shuffle each iteration:")
        l9.grid(row=9, column=0, sticky='e')
        self.shuf = tk.StringVar()
        self.shuf.set('True')
        dropmen9 = tk.OptionMenu(self.main_fr, self.shuf, 'True', 'False')
        dropmen9.configure(relief='groove')
        dropmen9.grid(row=9, column=1, sticky='nsew')

        l10 = tk.Label(self.main_fr, text="Random State:")
        l10.grid(row=10, column=0, sticky='e')
        self.e10 = tk.Entry(self.main_fr, width=10)
        self.e10.insert(1, 42)
        self.e10.grid(row=10, column=1, sticky='w')

        l11 = tk.Label(self.main_fr, text="Early stopping:")
        l11.grid(row=11, column=0, sticky='e')
        self.estop = tk.StringVar()
        self.estop.set('False')
        dropmen11 = tk.OptionMenu(self.main_fr, self.estop, 'True', 'False')
        dropmen11.configure(relief='groove')
        dropmen11.grid(row=11, column=1, sticky='nsew')

    def crosfit(self):
        try:
            hiddenly = [int(x) for x in self.e1.get().split(',') if x != '']
            if self.e5.get() =='auto':
                batchik = self.e5.get()
            else:
                batchik = int(self.e5.get())
            print(hiddenly)
            if self.shuf.get() == 'True':
                shuf = True
            else:
                shuf =False
                print('xd')

            if self.estop.get() == 'False':
                estopik = False
            else:
                estopik = True

            self.clf = MLPRegressor(hidden_layer_sizes = hiddenly,activation = self.Activation.get(),
                                         solver = self.solver.get(), alpha = float(self.e4.get()),
                                         batch_size = batchik, learning_rate = self.learning_rate.get(),
                                         learning_rate_init = float(self.e7.get()),max_iter = int(self.e8.get()),
                                         shuffle = shuf, random_state = int(self.e10.get()), early_stopping = estopik
                                         )

        except:
            messagebox.showinfo("Error", "Unknown format of parameters, please change it ")
        self.score_val()


class RandomReg(Base_Regress):
    def start(self):
        self.main_clf()
        self.controller.show_frame(RandomReg)
        l = tk.Label(self.main_fr, text = 'Parameters for Random Forest Regressor:',font = Big_Font)
        l.grid(row=0, column=0, columnspan=2, sticky='nsew', pady=8)

        l1 = tk.Label(self.main_fr, text = "The number of trees in the forest:")
        l1.grid(row =1,column =0,sticky ='e')
        self.e1 = tk.Entry(self.main_fr, width = 10)
        self.e1.insert(1,100)
        self.e1.grid(row =1,column =1,sticky ='w')

        l2 = tk.Label(self.main_fr, text = "Criteria:")
        l2.grid(row =2,column =0,sticky ='e')
        self.criteria = tk.StringVar()
        self.criteria.set(crit_reg[0])
        dropmen2 = tk.OptionMenu(self.main_fr, self.criteria, *crit_reg)
        dropmen2.configure(relief='groove')
        dropmen2.grid(row=2, column=1,sticky ='nsew')

        l3 = tk.Label(self.main_fr, text = "The maximum depth of tree:")
        l3.grid(row =3,column =0,sticky ='e')
        self.e3 = tk.Entry(self.main_fr, width = 10)
        self.e3.insert(1,"None")
        self.e3.grid(row =3,column =1,sticky ='w')

        l4 = tk.Label(self.main_fr, text = "The minimum number of sample to split:")
        l4.grid(row =4,column =0,sticky ='e')
        self.e4 = tk.Entry(self.main_fr, width = 10)
        self.e4.insert(1,2)
        self.e4.grid(row =4,column =1,sticky ='w')

        l5 = tk.Label(self.main_fr, text = "The minimum number of samples to be a leaf node")
        l5.grid(row =5,column =0,sticky ='e')
        self.e5 = tk.Entry(self.main_fr, width = 10)
        self.e5.insert(1,1)
        self.e5.grid(row =5,column =1,sticky ='w')

        l10 = tk.Label(self.main_fr, text = "Random State:")
        l10.grid(row =10,column =0,sticky ='e')
        self.e10 = tk.Entry(self.main_fr, width = 10)
        self.e10.insert(1,42)
        self.e10.grid(row =10,column =1,sticky ='w')

        l11 = tk.Label(self.main_fr, text = "Bootstrap:")
        l11.grid(row =11,column =0,sticky ='e')
        self.boot = tk.StringVar()
        self.boot.set('True')
        dropmen11 = tk.OptionMenu(self.main_fr, self.boot, 'True','False')
        dropmen11.configure(relief='groove')
        dropmen11.grid(row=11, column=1,sticky ='nsew')

    def crosfit(self):
        try:
            if self.e3.get() == "None":
                mdept = None
            else:
                mdept = int(self.e3.get())

            if self.boot.get() == 'False':
                boot = False
            else:
                boot = True

            self.clf = RandomForestRegressor(n_estimators=int(self.e1.get()), criterion=self.criteria.get(),
                                             max_depth=mdept,
                                             min_samples_split=int(self.e4.get()), min_samples_leaf=int(self.e5.get()),
                                             n_jobs=self.n_jobs, random_state=int(self.e10.get()), bootstrap=boot)

        except:
            messagebox.showinfo("Error", "Unknown format of parameters, please change it ")

        self.score_val()


class SVM_SVR(Base_Regress):
    def start(self):
        self.main_clf()
        self.controller.show_frame(SVM_SVR)
        l = tk.Label(self.main_fr, text='Parameters for SVM regressor:', font=Big_Font)
        l.grid(row=0, column=0, columnspan=2, sticky='nsew', pady=8)

        l1 = tk.Label(self.main_fr, text="Regularization parameter:")
        l1.grid(row=1, column=0, sticky='e')
        self.e1 = tk.Entry(self.main_fr, width=10)
        self.e1.insert(1, 1)
        self.e1.grid(row=1, column=1, sticky='w')

        l2 = tk.Label(self.main_fr, text="Kernel:")
        l2.grid(row=2, column=0, sticky='e')
        self.ker = tk.StringVar()
        self.ker.set(kernels[0])
        dropmen2 = tk.OptionMenu(self.main_fr, self.ker, *kernels)
        dropmen2.configure(relief='groove')
        dropmen2.grid(row=2, column=1, sticky='nsew')

        l3 = tk.Label(self.main_fr, text="Degree:")
        l3.grid(row=3, column=0, sticky='e')
        self.e3 = tk.Entry(self.main_fr, width=10)
        self.e3.insert(1, 3)
        self.e3.grid(row=3, column=1, sticky='w')

        l4 = tk.Label(self.main_fr, text="Gamma:")
        l4.grid(row=4, column=0, sticky='e')
        self.gamm = tk.StringVar()
        self.gamm.set(gammas[0])
        dropmen2 = tk.OptionMenu(self.main_fr, self.gamm, *gammas)
        dropmen2.configure(relief='groove')
        dropmen2.grid(row=4, column=1, sticky='nsew')

        l5 = tk.Label(self.main_fr, text="Maximum number of iterations ('-1' - no limit)")
        l5.grid(row=5, column=0, sticky='e')
        self.e5 = tk.Entry(self.main_fr, width=10)
        self.e5.insert(1, -1)
        self.e5.grid(row=5, column=1, sticky='w')

        l6 = tk.Label(self.main_fr, text="Epsilon")
        l6.grid(row=6, column=0, sticky='e')
        self.e6 = tk.Entry(self.main_fr, width=10)
        self.e6.insert(1, 0.1)
        self.e6.grid(row=6, column=1, sticky='w')

    def crosfit(self):

        self.clf = SVR(C=float(self.e1.get()), kernel=self.ker.get(), degree=int(self.e3.get()),
                       gamma=self.gamm.get(), max_iter=int(self.e5.get()), epsilon=float(self.e6.get()))

        self.score_val()


class LinReg(Base_Regress):
    def start(self):
        self.main_clf()
        self.controller.show_frame(LinReg)

        l = tk.Label(self.main_fr, text='Parameters for Linear regression:', font=Big_Font)
        l.grid(row=0, column=0,columnspan=2, sticky='nsew', pady=8)
        l5 = tk.Label(self.main_fr, text="Polynomial degree")
        l5.grid(row=5, column=0, sticky='e')
        self.e6 = tk.Entry(self.main_fr, width=10)
        self.e6.insert(1, 1)
        self.e6.grid(row=5, column=1, sticky='w')

    def crosfit(self):
        self.clf = PolynomialRegression(degree=int(self.e6.get()), n_jobs=self.n_jobs)
        self.score_val()

