import tkinter as tk
from tkinter.filedialog import askopenfilename

###Step 1: Create The App Frame
class AppFrame(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        ###call the parent constructor
        tk.Frame.__init__(self, parent, *args, **kwargs)

        self.parent = parent

        ###Create button
        btn = tk.Button(self, text='askopenfilename',command=self.askopenfilename)
        btn.pack(pady=5)

    def askopenfilename(self):
        ###ask filepath
        filepath = askopenfilename()

        ###if you selected a file path
        if filepath:
            ###add it to the filepath list
            self.parent.filepaths.append(filepath)

            ###put it on the screen
            lbl = tk.Label(self, text=filepath)
            lbl.pack()

###Step 2: Creating The App
class App(tk.Tk):
    def __init__(self, *args, **kwargs):
        ###call the parent constructor
        tk.Tk.__init__(self, *args, **kwargs)

        ###create filepath list
        self.filepaths = []

        ###show app frame
        self.appFrame = AppFrame(self)
        self.appFrame.pack(side="top",fill="both",expand=True)


###Step 3: Bootstrap the app
def main():
    app = App()
    app.mainloop()

if __name__ == '__main__':
    main()

