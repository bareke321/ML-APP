import pandas as pd
import numpy as np
import tkinter as tk

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
import multiprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelBinarizer, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from tkinter import messagebox
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, plot_confusion_matrix,classification_report, roc_curve
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sklearn.base import BaseEstimator, TransformerMixin
import seaborn as sns
sns.set(style="ticks")
import threading
import _thread
class MylabelBinarizer(TransformerMixin):
    def __init__(self,step=10, *args,**kwargs):
        self.encoder = LabelBinarizer(*args,**kwargs)
        self.step = step
    def fit(self,x,y=0):
        self.encoder.fit(np.array(x))
        self.classes_ = self.encoder.classes_
        return self
    def transform(self,x,y=0):
        X = []
        a = 0
        for wx in range(self.step, x.shape[0], self.step):
            X.append(x.iloc[a:wx].mode()[0])
            a = wx
        return self.encoder.transform(X)
    def inverse_transform(self,x,y=0):
        return self.encoder.inverse_transform(x)




class baseclassi(tk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent, relief='groove', borderwidth=20, bg='white')
        self.controller = controller
        self.keys = self.controller.frames_list
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=60)
        self.grid_rowconfigure(0, weight=1)
        self.controller.stan = 1
        self.controller.stan_reg =1
    def main_clf(self):
        for widget in self.winfo_children():
            widget.destroy()
        self.n_jobs = 1
        self.main_fr = tk.Frame(self, relief='groove', borderwidth=5)
        self.main_fr.grid(row=0, column=0, sticky='new')
        self.main_fr.grid_columnconfigure(0, weight=1)
        self.main_fr.grid_columnconfigure(1, weight=1)
        self.sec_fr = tk.Frame(self, relief='groove', borderwidth=5, bg='white')
        self.sec_fr.grid(row=0, column=1, sticky='nsew')
        self.sec_fr.grid_rowconfigure(0, weight=4)
        self.sec_fr.grid_rowconfigure(1, weight=1)
        self.sec_fr.grid_columnconfigure(0, weight=1)
        self.sec_fr.grid_columnconfigure(1, weight=1)


        self.b12 = tk.Button(self.main_fr, text = "Fit data into Cross validation Score",
                             relief='groove', command= lambda: _thread.start_new_thread(self.crosfit,()), height=3)
        self.b12.grid(row=13,column =0,columnspan = 2,sticky = 'nsew',pady =5)

        self.b14 = tk.Button(self.main_fr, text = "Check score on test set", relief='groove',
                        command=lambda: _thread.start_new_thread(self.testcheck,()), height = 3,state='disable')
        self.b14.grid(row=14,column =0,columnspan = 2,sticky = 'nsew',pady =5)

        self.b15 = tk.Button(self.main_fr, text = "Plot summary of test", relief='groove',
                        command=self.summr, height = 3,state='disable')
        self.b15.grid(row=15,column =0,columnspan = 2,sticky = 'nsew',pady =5)

        self.e15 = tk.Entry(self.main_fr, width = 25)
        self.e15.grid(row =16,column =0,sticky ='nse',padx=5)
        self.b16 = tk.Button(self.main_fr, text = "Save Classifier", command= self.save, relief='groove')
        self.b16.grid(row=16,column =1,sticky = 'nsew')
        scroll = tk.Scrollbar(self.sec_fr)
        scrollx = tk.Scrollbar(self, orient=tk.HORIZONTAL)
        scroll.grid(row=0, column=2, sticky='ns')
        scrollx.grid(row=2, column=1, columnspan=3, sticky='ew')

        self.txt = tk.Text(self.sec_fr, borderwidth=3, font=Big_Font, wrap=tk.NONE, yscrollcommand=scroll.set,
                           xscrollcommand=scrollx.set)
        self.txt.tag_configure("center", justify='center')
        self.txt.grid(row=0, column=0,columnspan=2, sticky='nsew')
        scroll.config(command=self.txt.yview)
        scrollx.config(command=self.txt.xview)
        self.f, self.axes = plt.subplots(1, 2)
        self.canvas = FigureCanvasTkAgg(self.f, self.sec_fr)
        self.canvas.get_tk_widget().grid(row=1, column=0, columnspan=2, sticky='nsew')
        self.canvas.draw()

    def join_thread(self):
        for thread in threading.enumerate():
            try:
                thread.join()
            except RuntimeError:
                pass

    def testcheck(self):
        self.b14['state'] = 'disable'
        self.clf.fit(self.controller.X_train,self.controller.y_train)
        scr = "Score:\n"+str(self.clf.score(self.controller.X_test,self.controller.y_test))+"\n\nConfusion Matrix\n"
        y_pred = self.controller.encoder.inverse_transform(self.clf.predict(self.controller.X_test))
        y_true =self.controller.encoder.inverse_transform(self.controller.y_test)
        self.conf = confusion_matrix(y_true, y_pred, labels=self.controller.encoder.classes_)
        class_report = classification_report(y_true, y_pred, target_names=[str(x) for x in self.controller.encoder.classes_])
        scr += str(pd.DataFrame(self.conf, columns=self.controller.encoder.classes_,
                                index=self.controller.encoder.classes_))
        scr+= '\n\n'+str(class_report)
        self.lab = list(self.controller.encoder.classes_)

        self.txt.delete('1.0', tk.END)
        self.txt.insert('1.0',scr)
        self.txt.tag_add('center', "1.0", tk.END)
        self.b14['state'] = 'normal'
        self.b15['state'] = 'normal'
        #self.join_thread()

    def summr(self):
        self.axes[1].clear()
        fpr = {}
        tpr = {}
        try:
            for i in range(len(self.lab)):
                fpr[i], tpr[i], _ = roc_curve(self.controller.y_test[:,i],self.clf.predict(self.controller.X_test)[:,i])
                self.axes[1].plot(fpr[i], tpr[i],label= self.lab[i])
        except IndexError:

            fpr, tpr, _ = roc_curve(self.controller.y_test, self.clf.predict(self.controller.X_test))
            self.axes[1].plot(fpr, tpr, label=self.lab)

        self.axes[1].set_ylabel("True Positive Rate")
        self.axes[1].set_xlabel("False Positive Rate")
        self.axes[1].legend(loc='best')
        self.axes[1].grid()

        self.axes[0].clear()
        #self.lab.insert(0, "spam")
        #self.axes[0].matshow(self.conf, cmap=plt.cm.gray)
        sns.heatmap(self.conf, square=True, annot=True, cbar=False,ax=self.axes[0],fmt='d')
        self.axes[0].set_xticklabels(labels=self.lab)
        self.axes[0].set_yticklabels(labels=self.lab)
        self.axes[0].set_ylabel("Actual class")
        self.axes[0].set_xlabel("Predicted class")
        #self.axes[0].xaxis.set_label_position('top')
        plt.tight_layout()
        self.f.tight_layout()
        self.canvas.draw()



    def save(self):


        self.controller.algho[self.e15.get()] = [self.clf, self.controller.processing_columns,
                                                   self.controller.full_pipe,self.controller.encoder,self.controller.ml_type]
        messagebox.showinfo("Success", "Classifier has been saved, good Job :)")
        self.e15.delete(0, 'end')

    def score_val(self):
        self.b12['state'] = 'disable'
        self.get_data(encoder=LabelBinarizer)
        scoval = cross_val_score(self.clf,self.controller.X_train, self.controller.y_train, cv=3,
                                     n_jobs =self.n_jobs,scoring = 'accuracy')


        self.txt.delete('1.0',tk.END)
        self.txt.insert('1.0',"Score from cross validation:"+str(scoval))
        self.b12['state'] = 'normal'
        self.b14['state'] = 'normal'
        #self.join_thread()

    def get_data(self, encoder= LabelEncoder):
        if self.controller.stan:
            if self.controller.ml_type !="Clustering":
                if self.controller.eval_dt:
                    self.controller.encoder = MylabelBinarizer(step=self.controller.step)
                else:
                    self.controller.encoder = encoder()
                if self.controller.ml_type == "Classification":
                    self.controller.target = self.controller.encoder.fit_transform(self.controller.df[self.controller.target_name])
                    self.controller.stan = 0
                elif self.controller.ml_type =="Regression":
                    self.controller.stan_reg = 0

                self.controller.fet_af = self.controller.full_pipe.fit_transform(self.controller.fet_set)
                self.controller.X_train, self.controller.X_test, self.controller.y_train, self.controller.y_test =\
                    train_test_split(self.controller.fet_af,self.controller.target,test_size = self.controller.test_size)
            else:
                self.controller.X_cl = self.controller.full_pipe.fit_transform(self.controller.fet_set)
        self.controller.stan = 0

class Mlp(baseclassi):
    def start(self):
        self.main_clf()
        #for widget, widget2 in zip(self.main_fr.winfo_children(),self.sec_fr.winfo_children()):
          #  widget.destroy()
           # widget2.destroy()
        self.controller.show_frame(Mlp)
        l = tk.Label(self.main_fr, text = 'Parameters for MLP classifier:',font = med_font)
        l.grid(row =0,column =0,columnspan=2,sticky ='nsew',pady = 5)

        l1 = tk.Label(self.main_fr, text = "Hidden layer sizes:",anchor='e')
        l1.grid(row =1,column =0,sticky ='nsew')
        self.e1 = tk.Entry(self.main_fr, width = 25)
        self.e1.insert(1,('100,'))
        self.e1.grid(row =1,column =1,sticky ='nsw')

        l2 = tk.Label(self.main_fr, text = "Activation function:", anchor='e')
        l2.grid(row =2,column =0,sticky ='nsew')
        self.Activation = tk.StringVar()
        self.Activation.set(actMLP[0])
        dropmen2 = tk.OptionMenu(self.main_fr, self.Activation, *actMLP)
        dropmen2.configure(relief='groove')
        dropmen2.grid(row=2, column=1,sticky ='nsew')

        l3 = tk.Label(self.main_fr, text = "Solver for weight optimization:", anchor='e')
        l3.grid(row =3,column =0,sticky ='nsew')
        self.solver = tk.StringVar()
        self.solver.set(solvMLP[0])
        dropmen3 = tk.OptionMenu(self.main_fr, self.solver, *solvMLP)
        dropmen3.configure(relief='groove')
        dropmen3.grid(row=3, column=1,sticky ='nsew')

        l4 = tk.Label(self.main_fr, text = "Alpha:", anchor='e')
        l4.grid(row =4,column =0,sticky ='nsew')
        self.e4 = tk.Entry(self.main_fr, width = 10)
        self.e4.insert(1,0.0001)
        self.e4.grid(row =4,column =1,sticky ='nsw')

        l5 = tk.Label(self.main_fr, text = "Batch size:", anchor='e')
        l5.grid(row =5,column =0,sticky ='nsew')
        self.e5 = tk.Entry(self.main_fr, width = 10)
        self.e5.insert(1,'auto')
        self.e5.grid(row =5,column =1,sticky ='nsw')

        l6 = tk.Label(self.main_fr, text = "Learning rate:", anchor='e')
        l6.grid(row =6,column =0,sticky ='nsew')
        self.learning_rate = tk.StringVar()
        self.learning_rate.set(learnrt[0])
        dropmen6 = tk.OptionMenu(self.main_fr, self.learning_rate, *learnrt)
        dropmen6.configure(relief='groove')
        dropmen6.grid(row=6, column=1,sticky ='nsew')

        l7 = tk.Label(self.main_fr, text = "Initial learning rate:", anchor='e')
        l7.grid(row =7,column =0,sticky ='nsew')
        self.e7 = tk.Entry(self.main_fr, width = 10)
        self.e7.insert(1,0.001)
        self.e7.grid(row =7,column =1,sticky ='nsw')

        l8 = tk.Label(self.main_fr, text = "Number of maximum iterations:", anchor='e')
        l8.grid(row =8,column =0,sticky ='e')
        self.e8 = tk.Entry(self.main_fr, width = 10)
        self.e8.insert(1,200)
        self.e8.grid(row =8,column =1,sticky ='nsw')

        l9 = tk.Label(self.main_fr, text = "Shuffle each iteration:",anchor='e')
        l9.grid(row =9,column =0,sticky ='nsew')
        self.shuf = tk.StringVar()
        self.shuf.set('True')
        dropmen9 = tk.OptionMenu(self.main_fr, self.shuf, 'True','False')
        dropmen9.configure(relief='groove')
        dropmen9.grid(row=9, column=1,sticky ='nsew')

        l10 = tk.Label(self.main_fr, text = "Random State:", anchor='e')
        l10.grid(row =10,column =0,sticky ='nsew')
        self.e10 = tk.Entry(self.main_fr, width = 10)
        self.e10.insert(1,42)
        self.e10.grid(row =10,column =1,sticky ='nsw')

        l11 = tk.Label(self.main_fr, text = "Early stopping:",anchor='e')
        l11.grid(row =11,column =0,sticky ='nsew')
        self.estop = tk.StringVar()
        self.estop.set('False')
        dropmen11 = tk.OptionMenu(self.main_fr, self.estop, 'True','False')
        dropmen11.configure(relief='groove')
        dropmen11.grid(row=11, column=1,sticky ='nsew')

    def crosfit(self):
        try:
            hiddenly = [int(x) for x in self.e1.get().split(',') if x != '']
            if self.e5.get() =='auto':
                batchik = self.e5.get()
            else:
                batchik = int(self.e5.get())

            if self.shuf.get() == 'True':
                shuf = True
            else:
                shuf =False

            if self.estop.get() == 'False':
                estopik = False
            else:
                estopik = True

            self.clf = MLPClassifier(hidden_layer_sizes = hiddenly,activation = self.Activation.get(),
                                         solver = self.solver.get(), alpha = float(self.e4.get()),
                                         batch_size = batchik, learning_rate = self.learning_rate.get(),
                                         learning_rate_init = float(self.e7.get()),max_iter = int(self.e8.get()),
                                         shuffle = shuf, random_state = int(self.e10.get()), early_stopping = estopik
                                         )
        except:
            messagebox.showinfo("Error", "Unknown format of parameters, please change it ")


        self.score_val()

class Knn(baseclassi):

    def start(self):
        self.main_clf()
        self.controller.show_frame(Knn)
        l = tk.Label(self.main_fr, text='Parameters for KKN classifier:', font=Big_Font)
        l.grid(row=0, column=0, columnspan=2, sticky='nsew', pady=15)

        l1 = tk.Label(self.main_fr, text="Number of neighbors:")
        l1.grid(row=1, column=0, sticky='e')
        self.e1 = tk.Entry(self.main_fr, width=25)
        self.e1.insert(1, 5)
        self.e1.grid(row=1, column=1, sticky='w')

        l2 = tk.Label(self.main_fr, text="Weights:")
        l2.grid(row=2, column=0, sticky='e')
        self.wei = tk.StringVar()
        self.wei.set('uniform')
        dropmen2 = tk.OptionMenu(self.main_fr, self.wei, *('uniform', 'distance'))
        dropmen2.configure(relief='groove')
        dropmen2.grid(row=2, column=1, sticky='nsew')

        l4 = tk.Label(self.main_fr, text="Algorithm:")
        l4.grid(row=4, column=0, sticky='e')
        self.kknalgo = tk.StringVar()
        self.kknalgo.set('auto')
        dropmen3 = tk.OptionMenu(self.main_fr, self.kknalgo, *('auto', 'ball_tree', 'kd_tree', 'brute'))
        dropmen3.configure(relief='groove')
        dropmen3.grid(row=4, column=1, sticky='nsew')

        l5 = tk.Label(self.main_fr, text="Power parameter")
        l5.grid(row=5, column=0, sticky='e')
        self.pow = tk.StringVar()
        self.pow.set('Euclidean distance')
        dropmen4 = tk.OptionMenu(self.main_fr, self.pow, *('Euclidean distance', 'Manhattan distance', 'Minkowski distance'))
        dropmen4.configure(relief='groove')
        dropmen4.grid(row=5, column=1, sticky='nsew')

    def crosfit(self):
        pp = 1
        if self.pow.get() == 'Euclidean distance':
            pp = 2
        elif self.pow.get() == 'Manhattan distance':
            pp = 1
        elif self.pow.get() == 'Minkowski distance':
            pp = 3

        self.clf = KNeighborsClassifier(n_neighbors=int(self.e1.get()), weights= self.wei.get(),
                                        algorithm=self.kknalgo.get(), p=pp, n_jobs=self.n_jobs)
        self.score_val()


class RandomForrest(baseclassi):
    def start(self):
        self.main_clf()
        self.controller.show_frame(RandomForrest)
        l = tk.Label(self.main_fr, text='Parameters for Random Forest:', font=Big_Font)
        l.grid(row=0, column=0,columnspan=2, sticky='nsew', pady=15)

        l1 = tk.Label(self.main_fr, text="The number of trees in the forest:")
        l1.grid(row=1, column=0, sticky='e')
        self.e1 = tk.Entry(self.main_fr, width=10)
        self.e1.insert(1, 100)
        self.e1.grid(row=1, column=1, sticky='w')

        l2 = tk.Label(self.main_fr, text="Criteria:")
        l2.grid(row=2, column=0, sticky='e')
        self.criteria = tk.StringVar()
        self.criteria.set(crit[0])
        dropmen2 = tk.OptionMenu(self.main_fr, self.criteria, *crit)
        dropmen2.configure(relief='groove')
        dropmen2.grid(row=2, column=1, sticky='nsew')

        l3 = tk.Label(self.main_fr, text="The maximum depth of tree:")
        l3.grid(row=3, column=0, sticky='e')
        self.e3 = tk.Entry(self.main_fr, width=10)
        self.e3.insert(1, "None")
        self.e3.grid(row=3, column=1, sticky='w')

        l4 = tk.Label(self.main_fr, text="The minimum number of sample to split:")
        l4.grid(row=4, column=0, sticky='e')
        self.e4 = tk.Entry(self.main_fr, width=10)
        self.e4.insert(1, 2)
        self.e4.grid(row=4, column=1, sticky='w')

        l5 = tk.Label(self.main_fr, text="The minimum number of samples to be a leaf node")
        l5.grid(row=5, column=0, sticky='e')
        self.e5 = tk.Entry(self.main_fr, width=10)
        self.e5.insert(1, 1)
        self.e5.grid(row=5, column=1, sticky='w')

        l10 = tk.Label(self.main_fr, text="Random State:")
        l10.grid(row=10, column=0, sticky='e')
        self.e10 = tk.Entry(self.main_fr, width=10)
        self.e10.insert(1, 42)
        self.e10.grid(row=10, column=1, sticky='w')

        l11 = tk.Label(self.main_fr, text="Bootstrap:")
        l11.grid(row=11, column=0, sticky='e')
        self.boot = tk.StringVar()
        self.boot.set('True')
        dropmen11 = tk.OptionMenu(self.main_fr, self.boot, 'True', 'False')
        dropmen11.configure(relief='groove')
        dropmen11.grid(row=11, column=1, sticky='nsew')

    def crosfit(self):

        if self.e3.get() == "None":
            mdept = None
        else:
            mdept = int(self.e3.get())

        if self.boot.get() == 'False':
            boot = False
        else:
            boot = True

        self.clf = RandomForestClassifier(n_estimators=int(self.e1.get()), criterion=self.criteria.get(),
                                          max_depth=mdept,
                                          min_samples_split=int(self.e4.get()), min_samples_leaf=int(self.e5.get()),
                                          n_jobs=self.n_jobs, random_state=int(self.e10.get()), bootstrap=boot)
        self.score_val()

class SVM_SVC(baseclassi):
    def start(self):
        self.main_clf()
        self.controller.show_frame(SVM_SVC)
        l = tk.Label(self.main_fr, text = 'Parameters for SVM:',font = Big_Font)
        l.grid(row =0, column =0, columnspan=2, sticky ='nsew', pady = 8)

        l1 = tk.Label(self.main_fr, text = "Regularization parameter:")
        l1.grid(row =1,column =0,sticky ='e')
        self.e1 = tk.Entry(self.main_fr, width = 10)
        self.e1.insert(1,1)
        self.e1.grid(row =1,column =1,sticky ='w')

        l2 = tk.Label(self.main_fr, text = "Kernel:")
        l2.grid(row =2,column =0,sticky ='e')
        self.ker = tk.StringVar()
        self.ker.set(kernels[0])
        dropmen2 = tk.OptionMenu(self.main_fr, self.ker, *kernels)
        dropmen2.configure(relief='groove')
        dropmen2.grid(row=2, column=1,sticky ='nsew')

        l3 = tk.Label(self.main_fr, text = "Degree:")
        l3.grid(row =3,column =0,sticky ='e')
        self.e3 = tk.Entry(self.main_fr, width = 10)
        self.e3.insert(1,3)
        self.e3.grid(row =3,column =1,sticky ='w')

        l4 = tk.Label(self.main_fr, text = "Gamma:")
        l4.grid(row =4,column =0,sticky ='e')
        self.gamm = tk.StringVar()
        self.gamm.set(gammas[0])
        dropmen3 = tk.OptionMenu(self.main_fr, self.gamm, *gammas)
        dropmen3.configure(relief='groove')
        dropmen3.grid(row=4, column=1,sticky ='nsew')

        l5 = tk.Label(self.main_fr, text = "Maximum number of iterations ('-1' - no limit)")
        l5.grid(row =5,column =0,sticky ='e')
        self.e5 = tk.Entry(self.main_fr, width = 10)
        self.e5.insert(1,-1)
        self.e5.grid(row =5,column =1,sticky ='w')

    def crosfit(self):
        self.clf = OneVsRestClassifier(SVC(C=float(self.e1.get()), kernel=self.ker.get(), degree=int(self.e3.get()),
                       gamma=self.gamm.get(), max_iter=int(self.e5.get())))
        self.score_val()
