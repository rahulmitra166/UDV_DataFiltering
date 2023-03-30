import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt

from DOPpy import *
from udv_analysis_lib import *

class UDV_GUI:
    def __init__(self, master):
        self.master = master
        master.title("UDV Processing GUI")

        # set up file selection widgets
        self.filepath = None
        self.file_label = tk.Label(master, text="No file selected")
        self.file_label.grid(row=0, column=0, columnspan=2, pady=5)
        self.select_button = tk.Button(master, text="Select File", command=self.select_file)
        self.select_button.grid(row=1, column=0, columnspan=2, pady=5)

        # set up processing parameter widgets
        self.threshold_label = tk.Label(master, text="Threshold Value:")
        self.threshold_label.grid(row=2, column=0, sticky='w', pady=5)
        self.threshold_entry = tk.Entry(master)
        self.threshold_entry.grid(row=2, column=1, pady=5)

        self.interpolation_label = tk.Label(master, text="Interpolation Method:")
        self.interpolation_label.grid(row=3, column=0, sticky='w', pady=5)
        self.interpolation_var = tk.StringVar(master)
        self.interpolation_var.set("linear") # default value
        self.interpolation_dropdown = tk.OptionMenu(master, self.interpolation_var, "none", "velo_max", "linear", "quadratic", "cubic")
        self.interpolation_dropdown.grid(row=3, column=1, pady=5)

        self.time_limits_label = tk.Label(master, text="Time Limits (tuple):")
        self.time_limits_label.grid(row=4, column=0, sticky='w', pady=5)
        self.time_limits_entry = tk.Entry(master)
        self.time_limits_entry.grid(row=4, column=1, pady=5)

        self.start_depth_label = tk.Label(master, text="Start Depth (mm):")
        self.start_depth_label.grid(row=5, column=0, sticky='w', pady=5)
        self.start_depth_entry = tk.Entry(master)
        self.start_depth_entry.grid(row=5, column=1, pady=5)

        self.depth_line_label = tk.Label(master, text="Depth Line (mm):")
        self.depth_line_label.grid(row=6, column=0, sticky='w', pady=5)
        self.depth_line_entry = tk.Entry(master)
        self.depth_line_entry.grid(row=6, column=1, pady=5)

        # set up save widgets
        self.save_plot_button = tk.Button(master, text="Save Plot", state=tk.DISABLED, command=self.save_plot)
        self.save_plot_button.grid(row=7, column=0, pady=5)

        self.save_data_button = tk.Button(master, text="Save Data", state=tk.DISABLED, command=self.save_data)
        self.save_data_button.grid(row=7, column=1, pady=5)

        # set default input values
        self.threshold_entry.insert(tk.END, "70.0")
        self.start_depth_entry.insert(tk.END, "50.0")
        self.time_limits_entry.insert(tk.END, "0, 20")
        self.depth_line_entry.insert(tk.END, "150.0")

        # set up button widgets
        self.process_button = tk.Button(master, text="Process Data", state=tk.NORMAL, command=self.process_data)
        self.process_button.grid(row=10, column=0, pady=10)

        self.refresh_button = tk.Button(master, text="Refresh", state=tk.NORMAL, command=self.refresh_gui)
        self.refresh_button.grid(row=10, column=1, pady=10)

        # initialize figure variables
        self.fig_raw = None
        self.fig = None

        # center the window on screen
        master.update_idletasks()
        w = master.winfo_width()
        h = master.winfo_height()
        extra_w = master.winfo_rootx() - master.winfo_x()
        extra_h = master.winfo_rooty() - master.winfo_y()
        x = (master.winfo_screenwidth() // 2) - (w // 2) - extra_w
        y = (master.winfo_screenheight() // 2) - (h // 2) - extra_h
        master.geometry("{}x{}+{}+{}".format(w, h, x, y))
        return
    
    def select_file(self):
        # open file dialog to select .BDD file
        filetypes = [("BDD Files", "*.BDD")]
        filepath = filedialog.askopenfilename(filetypes=filetypes)

        if filepath:
            self.filepath = filepath
            self.file_label.config(text=filepath)

    def process_data(self):
        if not hasattr(self, 'filepath'):
            messagebox.showerror("Error", "No file selected.")
            return

        # get processing parameters from input widgets
        try:
            thr = float(self.threshold_entry.get())
            ignore_depth = float(self.start_depth_entry.get())
            time_limit = self.time_limits_entry.get().split(",")
            time_limit = (int(time_limit[0]), int(time_limit[1]))
            depth_line = float(self.depth_line_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid input value.")
            return

        # read data from file
        try:
            bdd = DOP(self.filepath)
            depth = np.array(bdd.getDepth())[0]
            time = np.array(bdd.getTime())[0]
            data = np.array(bdd.getChannelParam('velo'))[0]
            data = data*1e3
            data = data.T
            raw_data = data.copy()
            print(data.shape)
        except:
            messagebox.showerror("Error", "Unable to read file.")
            return

        # ignore data up to specified depth
        s = np.searchsorted(depth, ignore_depth)
        # create UDV object and process data
        try:
            print("Processing data...")
            obj = UDV()
            print("Object created")
            fig_raw = obj.plot_raw_UDV(time, depth, raw_data, xlimits=(time_limit[0], time_limit[1]), levels=300)
            print("Raw fig created")
            corrected_data = obj.remove_outliers(time, depth, raw_data, start_id_depth=s, threshold=thr, interpolation_method=self.interpolation_var.get())
            print(corrected_data.shape)
            print("Outlier removed")
            fig_filtered = obj.plot_filtered_UDV(time, depth, corrected_data, xlimits=(time_limit[0], time_limit[1]), levels=300)
            print("Filt plotted")
            fig_line = obj.plot_lineplot(time, depth, raw_data, corrected_data, xlimits=(time_limit[0], time_limit[1]), depthLine = depth_line)
            print("Line plotted")
            fig_average = obj.plot_averageVz(depth, corrected_data, start_id_depth=s)
            print("Data processed successfully.")
        except:
            messagebox.showerror("Error", "Unable to process data.")
            return

        # update plot figures in GUI
        self.fig_raw = fig_raw
        self.fig_raw.show()
        self.fig_filtered = fig_filtered
        self.fig_filtered.show()
        self.fig_line = fig_line
        self.fig_line.show()
        self.fig_average = fig_average
        self.fig_average.show()

        # enable save buttons
        self.save_plot_button.config(state=tk.NORMAL)
        self.save_data_button.config(state=tk.NORMAL)
        self.obj = obj

    def save_plot(self):
        if self.fig_raw:
            # open file dialog to select directory and file name for saving
            filetypes = [("PNG Files", "*.png")]
            filepath = filedialog.asksaveasfilename(filetypes=filetypes, defaultextension=".png")

            if filepath:
                # save plots to specified file path
                if self.fig_raw:
                    self.fig_raw.savefig(filepath[:-4] + ".png", dpi=300, bbox_inches='tight')
                messagebox.showinfo("Save", "Plot files saved successfully.")
        else:
            messagebox.showerror("Error", "No plot available to save.")
        
        if self.fig_filtered:
            # open file dialog to select directory and file name for saving
            filetypes = [("PNG Files", "*.png")]
            filepath = filedialog.asksaveasfilename(filetypes=filetypes, defaultextension=".png")

            if filepath:
                # save plots to specified file path
                if self.fig_filtered:
                    self.fig_filtered.savefig(filepath[:-4] + ".png", dpi=300, bbox_inches='tight')
                messagebox.showinfo("Save", "Plot files saved successfully.")
        else:
            messagebox.showerror("Error", "No plot available to save.")

        if self.fig_line:
            # open file dialog to select directory and file name for saving
            filetypes = [("PNG Files", "*.png")]
            filepath = filedialog.asksaveasfilename(filetypes=filetypes, defaultextension=".png")

            if filepath:
                # save plots to specified file path
                if self.fig_line:
                    self.fig_line.savefig(filepath[:-4] + ".png", dpi=300, bbox_inches='tight')
                messagebox.showinfo("Save", "Plot files saved successfully.")
        else:
            messagebox.showerror("Error", "No plot available to save.")

        if self.fig_average:
            # open file dialog to select directory and file name for saving
            filetypes = [("PNG Files", "*.png")]
            filepath = filedialog.asksaveasfilename(filetypes=filetypes, defaultextension=".png")

            if filepath:
                # save plots to specified file path
                if self.fig_average:
                    self.fig_average.savefig(filepath[:-4] + ".png", dpi=300, bbox_inches='tight')
                messagebox.showinfo("Save", "Plot files saved successfully.")
        else:
            messagebox.showerror("Error", "No plot available to save.")

    def save_data(self):
        if hasattr(self, 'obj'):
            # open file dialog to select directory and file name for saving data
            filetypes = [("Data Files", "*.dat")]
            filepath = filedialog.asksaveasfilename(filetypes=filetypes, defaultextension=".dat")

            if filepath:
                # save processed data to specified file path
                try:
                    UDV.save_datafile(self.obj, filepath)
                except:
                    messagebox.showerror("Error", "Unable to save data.")
                    return

                messagebox.showinfo("Save", "Data saved successfully.")
        else:
            messagebox.showerror("Error", "No data available to save.")

    def refresh_gui(self):
        self.filepath = None
        self.file_label.config(text="No file selected")
        self.threshold_entry.delete(0, tk.END)
        self.threshold_entry.insert(tk.END, "2.0")
        self.interpolation_var.set("linear")
        self.time_limits_entry.delete(0, tk.END)
        self.time_limits_entry.insert(tk.END, "(0, 20)")
        self.start_depth_entry.delete(0, tk.END)
        self.start_depth_entry.insert(tk.END, "50")
        self.depth_line_entry.delete(0, tk.END)
        self.depth_line_entry.insert(tk.END, "150")
        self.save_plot_button.config(state=tk.DISABLED)
        self.save_data_button.config(state=tk.DISABLED)
        self.fig_raw = None
        self.fig = None
        self.obj = None


root = tk.Tk()
my_gui = UDV_GUI(root)
root.mainloop()
