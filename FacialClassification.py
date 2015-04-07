# -*- coding: utf-8 -*-
"""
Created on Sat Apr 04 23:49:03 2015

@author: Aishanee
"""
import pandas as pd
import numpy as np
import Tkinter
import os
from sys import argv
from PIL import Image



class ExpressionClassification(Tkinter.Tk):
    def __init__(self,parent,filename,start,end,destinationName):
        Tkinter.Tk.__init__(self,parent)
#        f = "D:\\Semester3\\StatDSP\\Project\\training\\training.csv"
        data = pd.read_csv(filename)
        self.dest = destinationName
        self.datapoints = int(end)-int(start)+1
        self.data = data[int(start):int(end)+1]
        self.parent = parent
        self.store_images = self.data.Image
        self.images = [[int(j) for j in i.split(' ')] for i in self.store_images]
        self.start = int(start)
        self.initialize()

    def initialize(self):
        self.grid()
        tp = Tkinter.Label(self,text="Select the appropriate emotional expression based on the image shown below")
        tp.grid(column=0,row=0,rowspan=2,columnspan=3 )
        self.imgnostring = Tkinter.StringVar()
        self.imgnostring.set("")
        self.imagenumberlabel = Tkinter.Label(self,textvariable = self.imgnostring)
        self.imagenumberlabel.grid(column = 0, row = 2)
        
        
        self.startButton = Tkinter.Button(self, text = "Start classifying",command = self.onStart)
        self.startButton.grid(column = 0, columnspan=3, row = 3, sticky = 'e')
        
        f = Image.fromarray(np.resize(self.images[0],(96,96)))
        f.save("yourIm.gif");
        ph = Tkinter.PhotoImage(file='yourIm.gif')
        self.imageLabel = Tkinter.Label(self,image=ph)
        self.imageLabel.label = ph
        self.imageLabel.grid(column=0,row=3,columnspan = 2,sticky="we")
        self.finalClassification = list()
        
        self.v = Tkinter.IntVar()
        self.v.set(1)  # initializing the choice
        self.expressions = [("Anger",1),("Fear",2),("Surprise",3),("Sadness",4),("Joy",5),("Disgust",6),("No Expression",7)]
        self.textlabel = Tkinter.Label(
              text = """Choose the correct expression""",
              justify = Tkinter.CENTER)
        self.textlabel.grid(column=0,row=4)
        def ShowChoice():
            return self.v.get
        x = 4
        for txt,val in self.expressions: 
            self.radioButton = Tkinter.Radiobutton(self,
                        text = txt,
                        variable = self.v,
                        command = ShowChoice,
                        value = val)
            self.radioButton.grid(column=0,row=x+1,sticky="W")
            x+=1
       

        self.nextButton = Tkinter.Button(self,text="Next",state = Tkinter.DISABLED,command = self.onNext)
        self.nextButton.grid(column=1,row=10)
        
        self.endButton = Tkinter.Button(self,text="Done",state = Tkinter.DISABLED, command = self.onEnd)
        self.endButton.grid(column=2,row=10)

        self.labelVariable = Tkinter.StringVar()
        self.label = Tkinter.Label(self,textvariable=self.labelVariable)
        self.label.grid(column=2,row=4)

        
    def onStart(self):
        self.labelVariable.set("")
        self.startButton.config(state = Tkinter.DISABLED )
        self.nextButton.config(state = Tkinter.ACTIVE)
        self.imageNo = 0
        self.imgnostring.set(self.imageNo + self.start)
        
    def onNext(self):
        self.finalClassification.append(self.v.get())
        self.imageNo += 1  
        if(self.imageNo == self.datapoints):
            self.nextButton.config(state = Tkinter.DISABLED)
            self.endButton.config(state = Tkinter.ACTIVE)
            return
        f = Image.fromarray(np.resize(self.images[self.imageNo],(96,96)))
        f.save("yourIm.gif");
        ph = Tkinter.PhotoImage(file='yourIm.gif')
        self.imageLabel.destroy()
        self.imageLabel=Tkinter.Label(self,image=ph)
        self.imageLabel.label = ph
        self.imageLabel.grid(column=0,row=3,columnspan=2,sticky="we")
        self.imgnostring.set(self.imageNo + self.start)
        self.v.set(1)
        
    def onEnd(self):
        self.data['classification_id'] = self.finalClassification
        f = open(self.dest,'wb')
        self.data.to_csv(f,index=False)
        print "New CSV file ",self.dest," is generated in the workding directory"
        self.destroy()
        
def main(argv):
    script, filename, record_start, record_end, dest = argv
    print "Running the Manual Classification GUI for ",script
    print "..."
    print "fetching records from ",record_start," to ",record_end
    print "..."
    app = ExpressionClassification(None,filename,record_start,record_end,dest)
    app.title('Classification of Facial Expression')
    app.mainloop()
    
    
if __name__ == "__main__":
    if len(argv) < 5:
        print "Enter in the formate : Script filename record_start record_end destinationFileName"
    main(argv)
