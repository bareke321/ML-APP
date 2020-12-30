import numpy as np
import pandas as pd
import tkinter as tk

class Create_Par(tk.Frame):
    def __init__(self, parent, controller ):
        super().__init__(parent, bg='white')
        self.controller = controller
        self.keys = self.controller.frames_list

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

    def create(self):
        for widget in self.winfo_children():
            widget.destroy()
        if type(self.controller.df) != str:
            self.controller.show_frame(Create_Par)

            self.main_fr = tk.Frame(self, relief='groove', borderwidth=5)
            self.main_fr.grid(row=0, column=0, sticky='new')
            self.main_fr.grid_columnconfigure(0, weight=1)
            self.main_fr.grid_columnconfigure(1, weight=1)
            self.main_fr.grid_columnconfigure(2, weight=1)
            #self.main_fr.grid_rowconfigure(0, weight=1)
            #self.main_fr.grid_rowconfigure(1, weight=1)
            #self.main_fr.grid_rowconfigure(2, weight=2)

            self.sec_fr = tk.Frame(self, relief='groove', borderwidth=5, bg='white')
            self.sec_fr.grid(row=0, column=1, sticky='nsew')
            self.sec_fr.grid_rowconfigure(0, weight=1)
            self.sec_fr.grid_columnconfigure(0, weight=1)

            l1 = tk.Label(self.main_fr,font=("Times New Roman", 13),
                          text= 'Please note that this '
                                'feature works only for columns which do\n not '
                                'have any special cases or "=" in name.\n'
                                ' Should any column has special cases or '
                                '\n"=" in the  name, please change it.'
                                '\n Example of use:\n'
                                'new_col = col_name_one / col_name_two')
            l1.grid(row=0,column=0, columnspan =3,sticky='nsew')

            scroll = tk.Scrollbar(self)
            scrollx = tk.Scrollbar(self, orient=tk.HORIZONTAL)
            scroll.grid(row=0, column=125, rowspan=50, sticky='ns')
            scrollx.grid(row=15, column=1, columnspan=160, sticky='ew')

            self.texthn = tk.Text(self.sec_fr, borderwidth=0,wrap=tk.NONE, yscrollcommand=scroll.set, xscrollcommand=scrollx.set)
            self.texthn.insert(tk.END, self.controller.df.head(50))
            self.texthn.grid(row=0,column=0,sticky='nsew')
            scroll.config(command=self.texthn.yview)
            scrollx.config(command=self.texthn.xview)

            self.textin = tk.Text(self.main_fr, wrap=tk.WORD,height=4,pady=6)
            self.textin.grid(row=1, column=0, columnspan =3, sticky="nsew")
            btk = tk.Button(self.main_fr, text="Submit",relief='groove',command=self.add_col)
            btk.grid(row=2,column=1, sticky="ew",pady=4)

            l2 = tk.Label(self.main_fr,text= "Change the name of column:", anchor='e')
            l2.grid(row=3,column=0,sticky="nsew")

            self.col = tk.StringVar()
            col_drop = tk.OptionMenu(self.main_fr, self.col, *self.controller.df.columns)
            col_drop.config(relief='groove')
            col_drop.grid(row=3, column=1, sticky='nsew', pady=2)

            self.col_name = tk.Entry(self.main_fr)
            self.col_name.insert(1, "New_Name")
            self.col_name.grid(row=3, column=2, sticky='nsew', pady=2)

            btk2 = tk.Button(self.main_fr, text="Apply",relief='groove', command=self.rename)
            btk2.grid(row=4, column=1, sticky='nsew')

            l3 = tk.Label(self.main_fr,text= "Choose column to delete:", anchor='e')
            l3.grid(row=5,column=0,sticky="nsew", pady=10)

            self.col_del = tk.StringVar()
            col_drop = tk.OptionMenu(self.main_fr, self.col_del, *self.controller.df.columns)
            col_drop.config(relief='groove')
            col_drop.grid(row=5, column=1, sticky='nsew', pady=10)

            btk3 = tk.Button(self.main_fr, text="Delete",relief='groove', command=self.delete)
            btk3.grid(row=5, column=2, sticky='nsew', pady=10)
        else:
            self.controller.error()
    def add_col(self):
        txt = self.textin.get("1.0",tk.END)
        txt_rob =txt.split("=")[1]
        for x in self.controller.df.columns:
            txt_rob = txt_rob.replace(x,f'self.controller.df["{x}"]')
        print(txt_rob)
        txt_rob2 = txt.split("=")[0].replace(txt.split('=')[0].strip(),f'self.controller.df["{txt.split("=")[0].strip()}"]')
        txt = "=".join([txt_rob2, txt_rob])
        print(txt)
        exec(txt)
        self.create()

    def delete(self):
        self.controller.df.drop(self.col_del.get(), axis=1, inplace=True)
        self.create()

    def rename(self):
        self.controller.df.rename(columns={self.col.get():self.col_name.get()}, inplace=True)
        self.create()
