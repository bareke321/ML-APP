import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import _thread

pd.set_option('display.max_columns',1000)
pd.set_option('display.width', 10000000)
pd.set_option('display.max_rows', 100000)

class Data_Load(tk.Frame):
    def __init__(self, parent, controller ):
        super().__init__(parent, bg='white')
        self.controller = controller
        self.keys()

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=12)
        for x in range(14):
            self.grid_rowconfigure(x, weight=1)
        #self.grid_rowconfigure(1, weight=1)
        #self.grid_rowconfigure(1, weight=1)

    def keys(self):
        self.keys = self.controller.frames_list
    def show_frame(self):
        self.controller.show_frame(Data_Load)

    def loading_file(self):
        for widget in self.winfo_children():
            widget.destroy()

        self.main_fr = tk.Frame(self,  relief='groove', borderwidth=5)
        self.main_fr.grid(row=0,column=0,sticky='new')

        self.sec_fr = tk.Frame(self,relief='groove', borderwidth=5,bg='white')
        self.sec_fr.grid(row=0,column=1,rowspan=14,sticky='nsew')
        self.sec_fr.grid_rowconfigure(0, weight=1)
        self.sec_fr.grid_columnconfigure(0, weight=1)
        self.show_frame()
        #self.controller.show_frame(Data_Load)
        self.sv = tk.StringVar()

        self.l1 = tk.Label(self.main_fr, text="Please indicate row from which data starts:")
        self.l1.grid(row=0, column=0, sticky='e')
        self.e1 = tk.Entry(self.main_fr, width=4, textvariable=self.sv,bg="#e8c5c5")
        self.sv.trace('w', self.callback)

        self.e1.grid(row=0, column=1, sticky='w')
        self.b1 = tk.Button(self.main_fr, text="Confirm",relief='groove',command=self.saving_df)
        self.b1.grid(row=10, column=0,pady=5)

        scroll = tk.Scrollbar(self)
        scrollx = tk.Scrollbar(self, orient=tk.HORIZONTAL)
        scroll.grid(row=0, column=125, rowspan=50, sticky='ns')
        scrollx.grid(row=50, column=1, columnspan=160, sticky='ew')
        self.l5 = tk.Text(self.sec_fr, borderwidth=0, yscrollcommand=scroll.set, xscrollcommand=scrollx.set,
                          wrap=tk.NONE)
        self.l5.tag_configure("highlight", background="#e8c5c5")
        self.l5.tag_configure("highlight_head", background="#e8e4c5")


        self.old ="3"
        scroll.config(command=self.l5.yview)
        scrollx.config(command=self.l5.xview)
        self.he_cl_old=""
        self.he_cl = tk.StringVar()
        self.he_cl.trace('w',self.call_head)
        self.l5.grid(row=0, column=0, padx=3, sticky='nsew')
        self.l4 = tk.Label(self.main_fr, text="Please indicate header row (if there is no header in file please set it to -1)")
        self.l4.grid(row=2, column=0, sticky='e')
        self.e4 = tk.Entry(self.main_fr, width=4,bg='#e8e4c5', textvariable=self.he_cl)

        self.e4.grid(row=2, column=1, sticky='w')
        self.l5.tag_add("highlight", "{}.0".format(3), "{}.end+1c".format(3))


        if ( self.filename.split('.')[-1].startswith('xl')):
            self.l3 = tk.Label(self.main_fr, text="Please set sheet number:")
            self.l3.grid(row=1, column=0, sticky='e')

            self.sh = tk.StringVar()

            self.e2 = tk.Entry(self.main_fr, width=4,textvariable=self.sh)
            self.e2.insert(1, 0)
            self.sh.trace('w', self.both)
            self.e2.grid(row=1, column=1, sticky='w')
            pd.read_excel(self.filename, header=None, nrows=11)
            self.start_head = []
            dff = pd.ExcelFile(self.filename)
            for amount in dff.sheet_names:
                try:
                    test = pd.read_excel(dff,header=None,sheet_name=amount,nrows=11)
                    self.l5.insert(tk.END,f" \n\nSheet name: {amount}:\n\n\n")
                    self.start_head.append(int(float(self.l5.index('end'))))
                    self.l5.insert(tk.END, str(test.head(10)))
                except IndexError:
                    break

        if ( self.filename.split('.')[-1].startswith('txt')):
            self.l3 = tk.Label(self.main_fr, text="Please indicate the separator for data:")
            self.l3.grid(row=1, column=0, sticky='e')
            self.e2 = tk.Entry(self.main_fr, width=4)
            self.e2.insert(1, ",")
            self.e2.grid(row=1, column=1, sticky='w')

            try:
                test = pd.read_csv(self.filename,header=None,nrows=11)
                self.l5.insert(tk.END, str(test.head(10)))
            except IndexError:
                    pass

        if (self.filename.split('.')[-1].startswith('csv')):
            self.l3 = tk.Label(self.main_fr, text="Please indicate the separator for data:")
            self.l3.grid(row=1, column=0, sticky='e')
            self.e2 = tk.Entry(self.main_fr, width=4)
            self.e2.insert(1, ",")
            self.e2.grid(row=1, column=1, sticky='w')

            try:
                test = pd.read_csv(self.filename,header=None,nrows=11)
                self.l5.insert(tk.END, str(test.head(10)))
            except IndexError:
                    pass
        self.e4.insert(1, 0)
        self.e1.insert(1, 1)



    def both(self,*args):
        if self.sh.get !="":
            self.callback(*args)
            self.call_head(*args)
    def call_head(self,*args):
        if self.filename.split('.')[-1].startswith('xl'):

            if self.he_cl.get() == '' or self.he_cl.get() == "-1" or self.he_cl.get() == "-" or self.sh.get() == "":
                self.l5.tag_remove("highlight_head", "{}.0".format(self.he_cl_old), "{}.end+1c".format(self.he_cl_old))

            else:
                n = int(self.sh.get())
                self.l5.tag_add("highlight_head", "{}.0".format(int(self.he_cl.get())+self.start_head[n]), "{}.end+1c".format(int(self.he_cl.get())+self.start_head[n]))
                self.he_cl_old = int(self.he_cl.get())+self.start_head[n]
        else:
            if self.he_cl.get() == '' or self.he_cl.get() == "-1" or self.he_cl.get() == "-":
                self.l5.tag_remove("highlight_head", "{}.0".format(self.he_cl_old), "{}.end+1c".format(self.he_cl_old))

            else:
                self.l5.tag_add("highlight_head", "{}.0".format(int(self.he_cl.get())+2), "{}.end+1c".format(int(self.he_cl.get())+2))
                self.he_cl_old = int(self.he_cl.get())+2

    def callback(self,*args):
        if self.filename.split('.')[-1].startswith('xl'):
            if self.sv.get() == '' or self.sh.get() == "":
                self.l5.tag_remove("highlight", "{}.0".format(self.old), "{}.end+1c".format(self.old))

            else:
                n = int(self.sh.get())
                self.l5.tag_add("highlight", "{}.0".format(int(self.sv.get())+self.start_head[n]), "{}.end+1c".format(int(self.sv.get())+self.start_head[n]))
                self.old = int(self.sv.get())+self.start_head[n]
        else:
            if self.sv.get() == '':
                self.l5.tag_remove("highlight", "{}.0".format(self.old), "{}.end+1c".format(self.old))

            else:
                self.l5.tag_add("highlight", "{}.0".format(int(self.sv.get())+2), "{}.end+1c".format(int(self.sv.get())+2))
                self.old = int(self.sv.get())+2


    def file_name(self):
        try:
            self.filename = filedialog.askopenfile(title="Select file").name

        except AttributeError:
            messagebox.showinfo("Error", "No file chosen")
        else:
            self.loading_file()

    def saving_df(self):
        if( self.filename.split('.')[-1].startswith('xl')):
            val_skip = int(self.e1.get())-1
            if val_skip <0:
                val_skip =0

            val_head = int(self.e4.get())
            val_sheet = int(self.e2.get())
            if val_head >0:
                val_skip = val_skip- val_head
            print(f"{val_head} - header")
            print(f"{val_skip} - od ktorego startujemy")

            skip = list(range(val_skip))
            if skip != [] and val_skip > val_head:
                skip.remove(val_head)
            print(skip)
            self.controller.df = pd.read_excel(self.filename,header= val_head,sheet_name=val_sheet)
            self.controller.df = self.controller.df.iloc[val_skip:]


        if(self.filename.split('.')[-1].startswith('csv')):

            val_skip = int(self.e1.get())-1
            val_head = int(self.e4.get())
            val_sep = self.e2.get()
            if val_skip <0:
                val_skip =0

            self.controller.df = pd.read_csv(self.filename,header=val_head,sep=val_sep)
            self.controller.df = self.controller.df.iloc[val_skip:]
            print(self.controller.df.head())

        if(self.filename.split('.')[-1].startswith('txt')):
            val_skip = int(self.e1.get())-1
            val_head = int(self.e4.get())
            val_sep = self.e2.get()

            self.controller.df = pd.read_csv(self.filename, header=val_head,sep=val_sep)
            self.controller.df = self.controller.df.iloc[val_skip:]


        self.controller.df.reset_index(drop=True,inplace=True)
        self.controller.columns = self.controller.df.columns
        for x in self.controller.df.columns:
            try:
                print('xd')
                self.controller.df[x] = pd.to_numeric(self.controller.df[x])
            except ValueError:
                pass
        abc = self.controller.df.isna().all(axis=0)
        abc = abc[abc==True].index

        self.controller.df.drop(abc,axis=1,inplace=True)
        self.controller.show_frame(self.keys[1])