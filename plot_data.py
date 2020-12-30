import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from matplotlib.colors import LogNorm
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import _thread
from PIL import ImageTk, Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks")


class Plot_Data(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.keys = self.controller.frames_list

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=12)
        self.grid_rowconfigure(0, weight=1)
        self.kind_of_plot = ["Line", "Scatter", "Histogram", "Box", "Swarm", "Heatmap", "PairPlot"]

    def inter(self):
        for widget in self.winfo_children():
            widget.destroy()
        if type(self.controller.df) != str:
            self.main_fr = tk.Frame(self,  relief='groove', borderwidth=5)
            self.main_fr.grid(row=0,column=0,sticky='nsew')

            self.sec_fr = tk.Frame(self,relief='groove', borderwidth=5)
            self.sec_fr.grid(row=0,column=1,sticky='nsew')

            self.main_fr.grid_columnconfigure(0, weight=1)

            self.sec_fr.grid_columnconfigure(0, weight=1)
            self.sec_fr.grid_rowconfigure(0, weight=1)
            self.controller.show_frame(Plot_Data)

            self.var_plot = tk.StringVar()
            self.var_plot.set("Kind of plot")
            #self.l1 = tk.Label(self.main_fr, text="Kind of plot")
            #self.l1.grid(row=0, column=0,sticky='nsew')
            self.drop_plot = tk.OptionMenu(self.main_fr,self.var_plot,*self.kind_of_plot)
            self.drop_plot.config(relief='groove')
            self.var_plot.trace("w", self.on_select)
            self.frame_on_select = tk.Frame(self.main_fr)
            self.frame_on_select.grid_columnconfigure(0, weight=1)
            self.frame_on_select.grid_columnconfigure(1, weight=1)
            self.frame_on_select.grid_rowconfigure(0, weight=1)

            self.frame_on_select.grid(row=1, column=0, sticky='nsew')


            self.drop_plot.grid(row=0, column=0, sticky='nsew',pady=5)

            self.but_plot = tk.Button(self.main_fr, text="Plot", command=self.plotting, relief='groove')
            self.but_plot.grid(row=2, column=0, sticky='nsew',pady=10)
        else:
            self.controller.error()

    def on_select(self,*args):

        for widget in self.frame_on_select.winfo_children():
            widget.destroy()

        if self.var_plot.get() in ("Line","Scatter") :
            self.x = tk.StringVar()
            self.x.set("X axis")

            self.y = tk.StringVar()
            self.y.set("Y axis")

            self.drop_x = tk.OptionMenu(self.frame_on_select, self.x, *self.controller.df.columns)
            self.drop_y = tk.OptionMenu(self.frame_on_select, self.y,*self.controller.df.columns)

            self.drop_x.config(relief='groove')
            self.drop_y.config(relief='groove')

            self.drop_x.grid(row=0,column=0, columnspan=2,sticky='nsew')
            self.drop_y.grid(row=1, column=0, columnspan=2, sticky='nsew')

            self.checkvar = tk.IntVar()
            self.checkx = tk.Checkbutton(self.frame_on_select, text ="Reverse X axis",variable = self.checkvar,onvalue = 1,offvalue =0 )
            self.checkx.grid(row=12, column=0, columnspan=2, sticky='nsew')

            self.checkvar2 = tk.IntVar()
            self.Chb2 = tk.Checkbutton(self.frame_on_select, text="Reverse Y axis", variable=self.checkvar2, onvalue=1, offvalue=0)
            self.Chb2.grid(row=13, column=0, columnspan=2, sticky='nsew')

            if self.var_plot.get() == "Scatter":
                lalp = tk.Label(self.frame_on_select,text="Alpha:",anchor='e')
                lalp.grid(row=2, column=0,sticky='nsew',pady=2)
                self.alp = tk.Entry(self.frame_on_select)
                self.alp.insert(1,1)
                self.alp.grid(row=2, column=1, sticky='nsew',pady=2)
                labS = tk.Label(self.frame_on_select, text='s :', anchor='e')
                labS.grid(row=3, column=0, sticky='nsew',pady=2)
                self.s = tk.StringVar()
                self.s.set(1)
                s_drop = tk.OptionMenu(self.frame_on_select, self.s, 1,5,10, *self.controller.df.columns)
                s_drop.config(relief='groove')
                s_drop.grid(row=3, column=1, sticky='nsew',pady=2)

                self.s_sign = tk.StringVar()
                self.s_sign.set("s /")
                s_sign_drop = tk.OptionMenu(self.frame_on_select,self.s_sign,"s /","s *")
                s_sign_drop.config(relief='groove')
                s_sign_drop.grid(row=4,column=0,sticky='nsew',pady=2)
                self.s_div = tk.Entry(self.frame_on_select)
                self.s_div.insert(1,1)
                self.s_div.grid(row=4,column=1,sticky='nsew',pady=2)

                labC = tk.Label(self.frame_on_select, text='c :', anchor='e')
                labC.grid(row=5,column=0,sticky='nsew',pady=2)
                self.c_sign = tk.StringVar()
                self.c_sign.set("None")
                c_sign_drop = tk.OptionMenu(self.frame_on_select,self.c_sign,"None", *self.controller.df.columns)
                c_sign_drop.config(relief='groove')
                c_sign_drop.grid(row=5,column=1,sticky='nsew',pady=2)

        if self.var_plot.get() in ("Histogram", "Box", "Swarm"):
            self.x = tk.StringVar()
            self.x.set("Axis")
            self.drop_x = tk.OptionMenu(self.frame_on_select, self.x, *self.controller.df.columns)
            self.drop_x.config(relief='groove')
            self.drop_x.grid(row=0, column=0, columnspan=2, sticky='nsew')
            if self.var_plot.get() == "Histogram":
                binL = tk.Label(self.frame_on_select,text="Number of bins:", anchor='e',pady=2)
                binL.grid(row=1,column=0, sticky='nsew')
                self.bins = tk.Entry(self.frame_on_select)
                self.bins.insert(1, "Default")
                self.bins.grid(row=1, column=1, sticky='nsew', pady=2)

                self.kde = tk.IntVar()
                self.kde_plt = tk.Checkbutton(self.frame_on_select, text ="Kde plot",variable = self.kde,onvalue = 1,offvalue =0 )
                self.kde_plt.grid(row=2, column=0, columnspan=2, sticky='nsew')

                self.rug = tk.IntVar()
                self.rug_plot = tk.Checkbutton(self.frame_on_select, text="Rug plot", variable=self.rug, onvalue=1, offvalue=0)
                self.rug_plot.grid(row=3, column=0, columnspan=2, sticky='nsew')

            if self.var_plot.get() in ("Box", "Swarm"):
                catL = tk.Label(self.frame_on_select,text="Categorical:",anchor='e')
                catL.grid(row=1, column=0, sticky='nsew', pady=2,)

                self.cat = tk.StringVar()
                self.cat.set("None")
                cat_drop = tk.OptionMenu(self.frame_on_select,self.cat,"None",*self.controller.df.columns)
                cat_drop.config(relief='groove')
                cat_drop.grid(row=1,column=1,sticky='nsew',pady=2)

                orient = tk.Label(self.frame_on_select,text="Orientation:",anchor='e')
                orient.grid(row=2, column=0, sticky='nsew', pady=2,)

                self.ori = tk.StringVar()
                self.ori.set("Vertical")
                ori_drop = tk.OptionMenu(self.frame_on_select,self.ori,"Vertical","Horizontal")
                ori_drop.config(relief='groove')
                ori_drop.grid(row=2,column=1,sticky='nsew',pady=2)

        if self.var_plot.get() == "Heatmap":
            catL = tk.Label(self.frame_on_select, text="Scale:", anchor='e')
            catL.grid(row=1, column=0, sticky='nsew', pady=2, )
            self.scale = tk.StringVar()
            self.scale.set("Linear")
            scale_drop = tk.OptionMenu(self.frame_on_select, self.scale, "Linear", "Logarithm")
            scale_drop.config(relief='groove')
            scale_drop.grid(row=1, column=1, sticky='nsew', pady=2)

        if self.var_plot.get() == "PairPlot":
            catL = tk.Label(self.frame_on_select, text="Hue:", anchor='e')
            catL.grid(row=1, column=0, sticky='nsew', pady=2, )
            self.hue = tk.StringVar()
            self.hue.set("None")
            scale_drop = tk.OptionMenu(self.frame_on_select, self.hue, "None",*self.controller.df.columns)
            scale_drop.config(relief='groove')
            scale_drop.grid(row=1, column=1, sticky='nsew', pady=2)





    def plotting(self):

        for widget in self.sec_fr.winfo_children():
            widget.destroy()

        self.but_plot['state'] = 'disable'
        self.f, self.axes = plt.subplots()

        if self.var_plot.get() not in ("Heatmap", "PairPlot"):

            self.canvas = FigureCanvasTkAgg(self.f, self.sec_fr)
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            #self.canvas.draw()
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.sec_fr)
            self.toolbar.update()
            self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self.axes.clear()
            x = self.x.get()
            if self.var_plot.get() in ("Line","Scatter"):

                y = self.y.get()
                if self.var_plot.get() =="Line":
                    self.axes.plot(self.controller.df[x],self.controller.df[y])
                else:
                    try:
                        if self.s_sign.get() == 's /':
                            fun =np.divide
                        else:
                            fun =np.multiply
                        if self.s.get() in ('1','5','10'):
                            s = int(self.s.get())
                            if self.c_sign.get() == "None":
                                self.controller.df.plot(kind="scatter", x=x, y=y, alpha=float(self.alp.get()),
                                                        s=fun(s,float(self.s_div.get())), ax=self.axes)
                            else:
                                self.controller.df.plot(kind="scatter", x=x, y=y, alpha=float(self.alp.get()),
                                                        s=fun(s,float(self.s_div.get())),label=s, ax=self.axes,
                                                        c=self.controller.df[self.c_sign.get()],cmap= plt.get_cmap("jet"),
                                                        colorbar = True)
                        else:
                            s = self.s.get()
                            if self.c_sign.get() == "None":
                                self.controller.df.plot(kind="scatter", x=x, y=y, alpha=float(self.alp.get()),
                                                        s=fun(self.controller.df[s],float(self.s_div.get())), ax=self.axes)
                            else:
                                self.controller.df.plot(kind="scatter", x=x, y=y, alpha=float(self.alp.get()),
                                                        s=fun(self.controller.df[s], float(self.s_div.get())),label=s,
                                                        ax=self.axes, c=self.controller.df[self.c_sign.get()],
                                                        cmap=plt.get_cmap("jet"), colorbar=True)
                    except:
                        self.axes.scatter(self.controller.df[x], self.controller.df[y])
                if self.checkvar2.get():
                    self.axes.invert_yaxis()

                if self.checkvar.get():
                    self.axes.invert_xaxis()

                self.axes.grid()
                self.axes.set(xlabel =x, ylabel =y)
                plt.legend()

            if self.var_plot.get() == "Histogram":
                if self.bins.get() =="Default":
                    bins = None
                else:
                    bins = int(self.bins.get())
                sns.distplot(self.controller.df[x], bins=bins, kde=self.kde.get(), rug=self.rug.get(), ax=self.axes)
                self.axes.set(xlabel=x)
            if self.var_plot.get() in ("Box", "Swarm"):
                if self.ori.get() == "Horizontal":
                    orient = 'h'
                else:
                    orient = 'v'

                if self.cat.get()=="None":
                    cat = None
                else:
                    cat = self.controller.df[self.cat.get()]
                    orient = 'h'
                if self.var_plot.get() == "Box":
                    sns.boxplot(x=self.controller.df[x], y=cat, orient=orient, ax=self.axes)
                else:
                    sns.swarmplot(x=self.controller.df[x], y=cat, orient=orient, ax=self.axes,s=5)
                self.axes.grid()

            plt.tight_layout()
            self.canvas.draw()
            self.toolbar.update()
        else:
            self.canvas = tk.Canvas(self.sec_fr)
            canvas = tk.Canvas(self.sec_fr)
            canvas.grid(row=0, column=0, sticky='nsew')

            yScrollbar = tk.Scrollbar(self)
            xScrollbar = tk.Scrollbar(self, orient='horizontal')
            yScrollbar.grid(row=0, column=125, rowspan=50, sticky='ns')
            xScrollbar.grid(row=15, column=1, columnspan=160, sticky='ew')
            canvas.config(yscrollcommand=yScrollbar.set)
            canvas.config(xscrollcommand=xScrollbar.set)
            yScrollbar.config(command=canvas.yview)
            xScrollbar.config(command=canvas.xview)
            self.f = plt.figure(dpi=100, figsize=(15, 10))
            self.axes = self.f.subplots()

            if self.var_plot.get() == "Heatmap":

                df = pd.DataFrame()
                for x in self.controller.df.columns:
                    if self.controller.df[x].dtype != object:
                        df[x]= self.controller.df[x]
                if self.scale.get() == "Linear":
                    sns.heatmap(df.dropna(), ax=self.axes)
                else:
                    for x in df.columns:
                        df[x] = df[x].loc[df[x] > 0]
                    abc = df.isna().all(axis=0)
                    abc = abc[abc == True].index
                    df.drop(abc, axis=1, inplace=True)

                    sns.heatmap(df.dropna(), ax=self.axes, norm=LogNorm())
                plt.tight_layout()
                self.f.savefig("heatmap_or_pairpolot.png")
            else:
                if self.hue.get() =="None":
                    hue = None
                else:
                    hue=self.hue.get()
                a = sns.pairplot(self.controller.df,hue=hue, plot_kws={"s": 15})
                for w in a.axes.flatten():
                    w.set_xlabel(w.get_xlabel(), rotation=45)
                    w.set_ylabel(w.get_ylabel(), rotation=0, labelpad=120)
                a.savefig("heatmap_or_pairpolot.png")

            self.im = ImageTk.PhotoImage(Image.open("heatmap_or_pairpolot.png"))
            self.imgtag = canvas.create_image(0, 0, anchor="nw", image=self.im)
            canvas.config(scrollregion=canvas.bbox(tk.ALL))
        self.but_plot['state'] = 'normal'