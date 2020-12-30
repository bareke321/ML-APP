from tkinter import *
from tkinter.ttk import *


#style = Style()
#style.configure("BW.TLabel", background='red')

main = Tk()
fl = Frame(main)
fl.pack(side="left", fill= "both", expand = True,padx=10)
fr = Frame(main)
bt = Button(main,text="-->")
bt.pack(side='left')
fr.pack(side="right", fill= "both", expand = True,padx=5,pady=5)
btr = Button(fr,text='he')
btr.pack(side='top',fill='x',pady=3,padx=3)
pow = StringVar()
dpr = OptionMenu(fr,pow,None,*(1,2,3))
dpr.pack(side='top',fill='x',pady=3,padx=3)
main.geometry("320x300")
main.mainloop()


def scatter_plot(self):
    for widget in self.sec_fr.winfo_children():
        widget.destroy()
    # self.f, self.axes = plt.subplots()
    # self.f, self.axes = plt.subplots()
    # self.axes.clear()
    # self.axes.plot([1,7],[6,3])
    self.f = Figure(dpi=1000, frameon=False)
    self.axes = self.f.subplots()
    self.axes.plot([1, 7], [6, 3])
    self.f.patch.set_facecolor('black')
    # scroll=tk.Scrollbar(self)
    scrollx = tk.Scrollbar(self, orient=tk.HORIZONTAL)
    # self.axes = self.f.add_subplot(111)
    self.canvas = FigureCanvasTkAgg(self.f, self.sec_fr)
    self.canvas.draw()
    self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')
    scroll = tk.Scrollbar(self)
    scroll.config(command=self.canvas.get_tk_widget().yview)
    scroll.grid(row=0, column=125, rowspan=50, sticky='ns')
    self.canvas.get_tk_widget().config(xscrollcommand=scrollx.set, yscrollcommand=scroll.set)

    # self.canvas.get_tk_widget().config(width=2500, height=2500)
    self.canvas.get_tk_widget().config(bg='red', scrollregion=(0, 0, 2500, 2500), confine=True)

    # self.toolbar = NavigationToolbar2Tk(self.canvas, self.sec_fr)
    # self.toolbar.update()
    # self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    # self.axes.clear()
    # self.axes.plot([1,7],[6,3])
    # ab = pd.plotting.scatter_matrix(self.controller.df,ax=self.axes,s=0.5)
    # for ax in ab.ravel():
    #    ax.set_xlabel(ax.get_xlabel(), fontsize=7, rotation=0)
    #    ax.set_ylabel(ax.get_ylabel(), fontsize=7, rotation=0, labelpad=15)

    # self.f.tight_layout()

    # self.toolbar.update()

    scrollx.config(command=self.canvas.get_tk_widget().xview)

    scrollx.grid(row=15, column=1, columnspan=160, sticky='ew')