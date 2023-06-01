from tkinter import Tk,ttk,messagebox,filedialog
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix

class Cancer:
    def __init__(self):
        self.root = Tk()
        self.root.title = "Used Car Price Prediction App"
        self.root.geometry('1000x600')
        
        self.lbl_fopen = ttk.Label(self.root,text="Open dataset:")
        self.lbl_fopen.grid(row=1,column=1)
        self.button_file_entry = ttk.Button(self.root,text="Select File",command=self.openFile)
        self.button_file_entry.grid(row=1,column=3)

        

        if __name__ == "__main__":
            self.root.mainloop()

    def openFile(self):
        try:
            self.file_path = filedialog.askopenfilename(filetypes=[("Excel files","*.xlsx"),("CSV files","*.csv")])
            f = self.file_path.split('/')[-1].split('.')
            filename = f[0]
            extension = f[1]
            self.df = pd.read_excel(self.file_path) if extension in ['xlsx','xls'] else pd.read_csv(self.file_path)
            print(self.df.columns,end="\n")
        except Exception as e:
            print(e)

        self.columns = [x for x in self.df.columns] 
        # print(self.columns)

        self.lbl_dependent = ttk.Label(self.root,text="Select Dependent Feature:")
        self.lbl_dependent.grid(row=2,column=1)

        self.combo_dependent = ttk.Combobox(self.root,values=self.columns)
        self.combo_dependent.grid(row=2,column=3)

        self.btn_combo_selected = ttk.Button(text="Select",command=self.delete_fields)
        self.btn_combo_selected.grid(row=2,column=5)

        
    def delete_fields(self):
        self.fields = []
        for num,name in enumerate(self.df.columns):
            self.fields.append(str(num)+":="+name)
        self.num_sep_col_list = ttk.Label(self.root,text="Number separated column list:")
        self.num_sep_col_list.grid(row=3,column=1)
        self.lbl_disp_fields = ttk.Combobox(self.root,values=self.fields)
        self.lbl_disp_fields.grid(row=3,column=3)
        self.lbl_ask_del_fields = ttk.Label(self.root,text="Enter comma-separated field numbers to delete")
        self.lbl_ask_del_fields.grid(row=4,column=1)
        self.del_entry = ttk.Entry(self.root)    
        self.del_entry.grid(row=4,column=3)
        self.btn_del_activate = ttk.Button(self.root,text="Delete",command=self.log_reg)
        self.btn_del_activate.grid(row=4,column=5)

    def log_reg(self):
        col_nums = self.del_entry.get().split(',')
        self.delete_list = [self.df.columns[int(x)] for x in col_nums]
        
        self.dependent_col_df = self.df[[self.combo_dependent.get()]]
        
        self.df.drop(self.delete_list,axis=1,inplace=True)
        

        self.independent_cols = list(set(self.df.columns) - set(self.dependent_col_df.columns))
        
        self.independent_col_df = self.df.loc[:,self.independent_cols]
        self.independent_col_df = pd.get_dummies(self.independent_col_df)        
        

        X = self.independent_col_df.values
        y = self.dependent_col_df.values

        
        
        # Encode dependent categorical variable
        le = LabelEncoder()
        y = le.fit_transform(y)
        
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

        s = StandardScaler()
        X_train = s.fit_transform(X_train)
        X_test = s.fit_transform(X_test)

        classifier = LogisticRegression(random_state=0)
        classifier.fit(X_train,y_train)

        y_pred = classifier.predict(X_test)
        print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

        print("_______:_______")
        cm = confusion_matrix(y_test,y_pred)
        print(cm)
        score = accuracy_score(y_test,y_pred)
        print(score)


Cancer()