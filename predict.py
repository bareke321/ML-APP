import tkinter as tk
import numpy as np
import pandas as pd
from io import StringIO
from tkinter import filedialog
from tkinter import messagebox
from data_load import Data_Load
from joblib import dump, load
import os

Big_Font = ("Times New Roman", 18)
med_font = ("Times New Roman",13)
pd.set_option('display.max_columns',1000)
pd.set_option('display.width', 10000000)
pd.set_option('display.max_rows', 100000)


class Prediction(tk.Frame):
    def __init__(self, parent, controller ):
        super().__init__(parent, bg='white')
        self.controller = controller
        self.keys = self.controller.frames_list

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

    def win_create(self):
        for widget in self.winfo_children():
            widget.destroy()
        self.controller.show_frame(Prediction)
        self.left_upper = tk.Frame(self,  relief='groove', borderwidth=5)
        self.left_upper.grid_columnconfigure(0, weight=1)
        self.left_upper.grid_columnconfigure(1, weight=1)
        self.left_upper.grid_rowconfigure(0, weight=1)
        self.left_upper.grid_rowconfigure(1, weight=1)
        self.left_upper.grid_rowconfigure(2, weight=3)
        self.left_upper.grid(row=0,column=0,sticky='nsew')

        self.right_upper = tk.Frame(self,  relief='groove', borderwidth=5)
        self.right_upper.grid(row=0,column=1,sticky='nsew')
        self.right_upper.grid_columnconfigure(0, weight=1)
        self.right_upper.grid_rowconfigure(0, weight=1)

        self.left_lower = tk.Frame(self,  relief='groove', borderwidth=5)
        self.left_lower.grid_columnconfigure(0,weight=1)
        self.left_lower.grid_columnconfigure(1, weight=1)
        self.left_lower.grid(row=1,column=0,sticky='nsew')

        self.right_lower = tk.Frame(self,  relief='groove', borderwidth=5)
        self.right_lower.grid(row=1,column=1,sticky='nsew')
        self.right_lower.grid_columnconfigure(0, weight=1)
        self.right_lower.grid_rowconfigure(0, weight=1)

        lx = tk.Label(self.left_upper, text='Algorithm info:', font=Big_Font)
        lx.grid(row=0,column=0, columnspan=2, sticky='nsew',pady=3)
        l1 = tk.Label(self.left_upper, text="Please choose your saved classifier/regressor:")
        l1.grid(row=1, column=0, sticky='e',pady=2)
        self.algh = tk.StringVar()
        #self.algh.set(list(self.controller.algho.keys())[0])
        self.algh.trace('w',self.on_alg_change)
        self.dropmen2 = tk.OptionMenu(self.left_upper, self.algh, *self.controller.algho.keys())
        self.dropmen2.configure(relief='groove')
        self.dropmen2.grid(row=1, column=1, sticky='nsew',pady=2)
        scroll_left_upper_x = tk.Scrollbar(self.left_upper,orient=tk.HORIZONTAL)
        scroll_left_upper_x.grid(row=3,column=0,columnspan=2,sticky='ew')
        self.text_alg = tk.Text(self.left_upper, wrap=tk.NONE,xscrollcommand=scroll_left_upper_x.set,height=19)
        self.text_alg.grid(row=3, column=0, columnspan=2, sticky="nsew",pady=2)
        tk.Button(self.left_upper,text="Save algorithm to file",relief='groove',command=self.save_model).grid(row=2,column=0,sticky='nsew')
        tk.Button(self.left_upper, text="Load algorithm from file",relief='groove',command=self.new_alg).grid(row=2, column=1, sticky='nsew')
        scroll_left_upper_x.config(command=self.text_alg.xview)

        scroll_right_upper = tk.Scrollbar(self.right_upper)
        scroll_right_upper.grid(row=0,rowspan=2,column=1,sticky='ns')
        scroll_right_upper_x = tk.Scrollbar(self.right_upper, orient=tk.HORIZONTAL)
        scroll_right_upper_x.grid(row=1,column=0,sticky='ew')
        self.text_to_predict = tk.Text(self.right_upper, wrap=tk.NONE,xscrollcommand=scroll_right_upper_x.set,
                                       yscrollcommand=scroll_right_upper.set)
        scroll_right_upper.config(command=self.text_to_predict.yview)
        scroll_right_upper_x.config(command=self.text_to_predict.xview)
        self.text_to_predict.grid(row=0,column=0, sticky='nsew')
        btk_pred = tk.Button(self.right_upper, text="Predict (PLEASE USE ';' AS SEPARATOR)",relief='groove', command=self.predict_rows)
        btk_pred.grid(row=2,column=0,sticky='nsew')
        btk_clean = tk.Button(self.right_upper,text='Clean', relief='groove',command=self.on_alg_change)
        btk_clean.grid(row=3, column=0,sticky='nsew')

        ll = tk.Label(self.left_lower, text='Predict new data from file:',font=med_font)
        ll.grid(row=0,column=0,columnspan=2,sticky='nsew',pady=6)
        l5 = tk.Label(self.left_lower, text= "Please choose file to load a new data for predictions:")
        l5.grid(row=1,column=0, sticky='e')
        btk_load = tk.Button(self.left_lower, text='Load',relief='groove', command=self.predict_file)
        btk_load.grid(row=1,column=1,sticky='nsew')
        l6 = tk.Label(self.left_lower,text = "Click to save predictions file:")
        l6.grid(row=2, column=0, sticky = 'e',pady=5)
        b15 = tk.Button(self.left_lower, text = "Save as",relief='groov', command=self.save_as)
        b15.grid(row=2,column =1,sticky='nsew',pady=5)

        scroll_right_lower = tk.Scrollbar(self.right_lower)
        scroll_right_lower.grid(row=0,column=1,sticky='ns')
        scroll_right_lower_x = tk.Scrollbar(self.right_lower, orient=tk.HORIZONTAL)
        scroll_right_lower_x.grid(row=1,column=0,sticky='ew')
        self.new_data_txt = tk.Text(self.right_lower,wrap=tk.NONE,xscrollcommand=scroll_right_lower_x.set,
                                    yscrollcommand=scroll_right_lower.set)
        scroll_right_lower.config(command=self.new_data_txt.yview)
        scroll_right_lower_x.config(command=self.new_data_txt.xview)
        self.new_data_txt.grid(row=0, column=0,sticky='nsew')

    def save_model(self):
        filename = filedialog.asksaveasfilename(initialdir="/", title="abc")
        dump(self.controller.algho[self.algh.get()],".".join([filename,'joblib']))
        messagebox.showinfo("Success", f"Model has been saved on {filename}")


    def new_alg(self):
        self.filename = filedialog.askopenfile(title="Select file").name

        self.controller.algho[os.path.basename(self.filename).split('.')[0]] = load(self.filename)
        self.dropmen2['menu'].add_command(label=os.path.basename(self.filename).split('.')[0])
        self.algh.set(os.path.basename(self.filename).split('.')[0])
    def on_alg_change(self,*args):
        if self.algh.get() =="-":
            pass
        else:
            self.text_to_predict.delete('1.0', tk.END)
            in_col = ';'.join(self.controller.algho[self.algh.get()][1])
            self.text_to_predict.insert(tk.END, in_col)
            self.text_to_predict.insert(tk.END,'\n')
            self.text_alg.delete('1.0', tk.END)
            strc = "Data Info:\nFeature Columns: "+ ", ".join(self.controller.algho[self.algh.get()][1])+"\n"
            #if self.controller.ml_type != "Clustering":
                #strc += "Target Column:" + self.controller.target_name + "\n\n"
            strc += "Pipeline Info:\n" + str(self.controller.algho[self.algh.get()][2]) +"\n\n"
            strc +="Algorithm Info:\n"+ str(self.controller.algho[self.algh.get()][0])
            self.text_alg.insert(tk.END, strc)


    def predict_rows(self):
        fet = self.text_to_predict.get('1.0',tk.END)
        df = pd.read_csv(StringIO(fet), sep=';')
        print(df.head())
        value_to_pred = self.controller.algho[self.algh.get()][2].transform(df)
        y_pred = self.controller.algho[self.algh.get()][0].predict(value_to_pred)
        if self.controller.algho[self.algh.get()][4] == "Classification":
            df['Predicted Value'] = self.controller.algho[self.algh.get()][3].inverse_transform(y_pred)
        else:
            df['Predicted Value'] = y_pred
        self.new_data_txt.delete('1.0',tk.END)
        self.new_data_txt.insert(tk.END, df)

    def predict_file(self):
            self.filename = filedialog.askopenfile(title="Select file").name
            self.load_data_win()
            """
            self.ab = tk.Tk()
            fr = tk.Frame(self.ab)
            fr.grid()
            l1 = tk.Label(fr, text="Please indicate row from which data starts:")
            l1.grid(row=0, column=0, sticky='w')
            self.e1 = tk.Entry(fr, width=4)
            self.e1.insert(1, 1)
            self.e1.grid(row=0, column=1, sticky='e')
            b1 = tk.Button(fr, text="Confirm", command=self.load_data)
            b1.grid(row=10, column=0)
            l4 = tk.Label(fr, text="Please indicate header row (if there is no header in file please set it to -1)")
            l4.grid(row=2, column=0, sticky='w')
            self.e4 = tk.Entry(fr, width=4)
            self.e4.insert(1, 0)
            self.e4.grid(row=2, column=1, sticky='e')

            if ( self.filename.split('.')[-1].startswith('xl')):
                l3 = tk.Label(fr, text="Please set sheet number:")
                l3.grid(row=1, column=0, sticky='w')
                self.e2 = tk.Entry(fr, width=4)
                self.e2.insert(1, 0)
                self.e2.grid(row=1, column=1, sticky='e')

            if ( self.filename.split('.')[-1].startswith('txt')):
                l3 = tk.Label(fr, text="Please indicate the separator for data:")
                l3.grid(row=1, column=0, sticky='w')
                self.e2 = tk.Entry(fr, width=4)
                self.e2.insert(1, ";")
                self.e2.grid(row=1, column=1, sticky='e')


            if ( self.filename.split('.')[-1].startswith('csv')):
                l3 = tk.Label(fr, text="Please indicate the separator for data:")
                l3.grid(row=1, column=0, sticky='w')
                self.e2 = tk.Entry(fr, width=4)
                self.e2.insert(1, ",")
                self.e2.grid(row=1, column=1, sticky='e')

            self.ab.mainloop()
        """
    def load_data_win(self):
        self.load_win = tk.Toplevel()
        self.fr_load = New_Data(self.load_win,self)
        self.fr_load.pack(fill='both', expand='True')
        self.fr_load.file_read()
        self.fr_load.loading_file()

    def load_data(self):
        """
        if (self.filename.split('.')[-1].startswith('xl')):
            val_skip = int(self.e1.get()) - 1
            if val_skip < 0:
                val_skip = 0

            val_head = int(self.e4.get())
            val_sheet = int(self.e2.get())
            if val_head > 0:
                val_skip = val_skip - val_head

            skip = list(range(val_skip))
            if skip != [] and val_skip > val_head:
                skip.remove(val_head)
            print(skip)
            self.df_test = pd.read_excel(self.filename, header=val_head, sheet_name=val_sheet)
            self.df_test = self.df_test.iloc[val_skip:]

        if self.filename.split('.')[-1].startswith('csv'):
            val_skip = int(self.e1.get()) - 1
            val_head = int(self.e4.get())
            val_sep = self.e2.get()
            if val_skip < 0:
                val_skip = 0

            self.df_test = pd.read_csv(self.filename, header=val_head, sep=val_sep)
            self.df_test = self.df_test.iloc[val_skip:]


        if (self.filename.split('.')[-1].startswith('txt')):
            val_skip = int(self.e1.get()) - 1
            val_head = int(self.e4.get())
            val_sep = self.e2.get()
            self.df_test = pd.read_csv(self.filename, header=val_head, sep=val_sep)
            self.df_test = self.df_test.iloc[val_skip:]
        """
        try:
            self.df_test = self.df_test[self.controller.algho[self.algh.get()][1]]
            for x in self.df_test.columns:
                try:
                    self.df_test[x] = pd.to_numeric(self.df_test[x])
                except ValueError:
                    pass
            #self.ab.destroy()
            self.load_win.destroy()
            value_to_pred = self.controller.algho[self.algh.get()][2].transform(self.df_test)
            y_pred = self.controller.algho[self.algh.get()][0].predict(value_to_pred)
            if self.controller.algho[self.algh.get()][4] == "Classification":
                if self.controller.eval_dt:
                    a = []
                    for x in self.controller.algho[self.algh.get()][3].inverse_transform(y_pred):
                        for y in range(self.controller.step):
                            a.append(x)
                    for x in range(self.df_test.shape[0]-len(a)):
                        a.append("Not known")
                    self.df_test['Predicted Value'] = a
                else:
                    self.df_test['Predicted Value'] = self.controller.algho[self.algh.get()][3].inverse_transform(y_pred)
            else:
                self.df_test['Predicted Value'] = y_pred

            self.new_data_txt.delete('1.0', tk.END)
            self.new_data_txt.insert(tk.END, self.df_test)
        except ValueError as a:
            messagebox.showinfo("Error", a)
    def save_as(self):
        filename = filedialog.asksaveasfilename(initialdir = "/",title ="abc",filetypes = (("csv files","*.csv"),("all files","*.*")))
        self.df_test.to_csv(".".join([filename,'csv']), index=False)
        messagebox.showinfo("Success", f"File has been saved on {filename}")

class New_Data(Data_Load):
    def __init__(self,parent,controller):
        super().__init__(parent,controller)
    def keys(self):
        pass
    def file_read(self):
        self.filename = self.controller.filename
    def show_frame(self):
        pass
    def saving_df(self):
        if (self.filename.split('.')[-1].startswith('xl')):
            val_skip = int(self.e1.get()) - 1
            if val_skip < 0:
                val_skip = 0

            val_head = int(self.e4.get())
            val_sheet = int(self.e2.get())
            if val_head > 0:
                val_skip = val_skip - val_head

            skip = list(range(val_skip))
            if skip != [] and val_skip > val_head:
                skip.remove(val_head)
            print(skip)
            self.df_test = pd.read_excel(self.filename, header=val_head, sheet_name=val_sheet)
            self.df_test = self.df_test.iloc[val_skip:]

        if self.filename.split('.')[-1].startswith('csv'):
            val_skip = int(self.e1.get()) - 1
            val_head = int(self.e4.get())
            val_sep = self.e2.get()
            if val_skip < 0:
                val_skip = 0

            self.df_test = pd.read_csv(self.filename, header=val_head, sep=val_sep)
            self.df_test = self.df_test.iloc[val_skip:]


        if (self.filename.split('.')[-1].startswith('txt')):
            val_skip = int(self.e1.get()) - 1
            val_head = int(self.e4.get())
            val_sep = self.e2.get()
            self.df_test = pd.read_csv(self.filename, header=val_head, sep=val_sep)
            self.df_test = self.df_test.iloc[val_skip:]
        self.controller.df_test = self.df_test
        self.controller.load_data()