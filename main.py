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
        self.root.geometry('600x600')
        
        self.lbl_fopen = ttk.Label(self.root,text="Open dataset:")
        self.lbl_fopen.grid(row=1,column=1)
        self.button_file_entry = ttk.Button(self.root,text="Select File",command=self.openFile)
        self.button_file_entry.grid(row=1,column=3)

        self.columns = [x for x in self.df.columns] 

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
        
        self.generate_input_data_fields()

    
    def generate_input_data_fields(self):
        pass
        # self.labels

    def log_reg(self):
        
        X = self.df.iloc[:,2:].values
        y = self.df.iloc[:,1].values
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

        # cm = confusion_matrix(y_test,y_pred)
        # print(cm)
        # score = accuracy_score(y_test,y_pred)
        # print(score)


Cancer()