import numpy as np
from scipy.interpolate import CubicSpline, UnivariateSpline, InterpolatedUnivariateSpline, interp1d
import matplotlib.pyplot as plt

class UDV:
    def __init__(self):
        return
    def detect_outliers(self, data, threshold):
        """
        Arguments
        ---------
        
        data --> 1D array
        threshold --> threshold value for the derivative
        
        Return
        ------
        
        bool_array --> boolean 1D array with True indices where an outlier is detected.
        """
        
        bool_array = np.full(len(data), False)
        change = np.diff(data)
        id_jumps = np.where(np.absolute(change)>threshold)[0]
        for idx in range (0,len(id_jumps) -1, 2):
            idx_s = id_jumps[idx]
            idx_e = id_jumps[idx+1]+1
            bool_array[idx_s:idx_e] = True
        return bool_array
    
    def remove_outliers(self, time, depth, raw_data, start_id_depth = 0, threshold = 70.0, interpolation_method = "linear"):
        """
        Arguments
        ---------
        
        start_id_depth --> values before start_id_depth will be ignored
        threshold --> threshold value for the derivative
        interpolation_method --> type of interpolation for outliers
        
        Return
        ------
        
        udv_data --> corrected 2D UDV data
        """
        corrected_data = raw_data.copy()
        for t,time in enumerate(time):
            #print("%d\t%.5f" %(t, time))
            data = raw_data[start_id_depth:-4,t]
            depth = depth[start_id_depth:-4]
            is_outlier = self.detect_outliers(data.copy(), threshold)
            idx = np.where(is_outlier==True)[0]
            data[idx] = np.nan
            if(interpolation_method == "none"):
                interpolated_data = data
            elif(interpolation_method == "velo_max"):
                max_val = np.nanmax(data)
                min_val = np.nanmin(data)
                if(abs(min_val)>abs(max_val)):
                    d = min_val
                else:
                    d = max_val
                data[idx] = d
                interpolated_data = data
            else:
                interpolated_data = self.interpolation(data, interpolation_method)
            corrected_data[start_id_depth:-4,t] = interpolated_data
        return corrected_data
    
    def interpolation(self, data, interpolation_method = "linear"):
        not_nan_indices = np.where(~np.isnan(data))[0]
        interpolation_func = interp1d(not_nan_indices, data[not_nan_indices], kind=interpolation_method)
        interpolated_data = interpolation_func(np.arange(len(data)))
        return interpolated_data
    
    def plot_raw_UDV(self, time, depth, data, xlimits, levels=300):
        fig1 = plt.figure(1)
        plt.clf()
        plt.gcf().set_size_inches([16,4])

        plt.contourf(time, depth, data, levels, cmap='viridis')
        plt.xlim(xlimits)
        cb = plt.colorbar()
        cb.set_ticks([data.min(), data.min()+((data.max()-data.min())/2), data.max()])
        plt.ylabel('Raw')
        plt.xlabel("Time (s)")
        plt.tight_layout()

        return fig1
    
    def plot_filtered_UDV(self, time, depth, data, xlimits, levels=300):
        fig2 = plt.figure(2)
        plt.clf()
        plt.gcf().set_size_inches([16,4])

        plt.contourf(time, depth, data, levels, cmap='viridis')
        plt.xlim(xlimits)
        cb = plt.colorbar()
        cb.set_ticks([data.min(), data.min()+((data.max()-data.min())/2), data.max()])
        plt.ylabel('Filtered')
        plt.xlabel("Time (s)")
        plt.tight_layout()

        return fig2

    def plot_lineplot(self, time, depth, raw_data, data, xlimits, depthLine = 150):

        fig3 = plt.figure(3)
        plt.clf()
        plt.gcf().set_size_inches([16,4])

        plt.plot(time, raw_data[np.searchsorted(depth, depthLine), :], "-r", label="Raw")
        plt.plot(time, data[np.searchsorted(depth, depthLine), :], "-b", label="Filt.")
        plt.xlim(xlimits)
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (mm/s)")
        plt.legend()
        plt.tight_layout()

        return fig3
    
    def plot_averageVz(self, depth, data, start_id_depth = 30):
        average_velo = np.nanmean(data, axis=1)
        fig4 = plt.figure(4)
        plt.clf()
        plt.gcf().set_size_inches([16,4])

        plt.plot(depth[start_id_depth:], average_velo[start_id_depth:], "-k", label="Mean vz")
        plt.xlabel("Average Vz (mm/s)")
        plt.ylabel("Depth (mm)")
        plt.tight_layout()

        return fig4
    
    def plotUDV(self, time, depth, raw_data, data, xlimits, start_id_depth=50, depthLine=150, levels=300):
        """
        Arguments
        ---------
        depthLine: int or float
            Depth at which lineplot is to be plotted
        xlimits: tuple
            X limits of plot
        levels: int
            Number of colorbar levels
        """
        avg_velo = np.nanmean(data, axis=1)
        
        fig, ax = plt.subplots(4, 1, figsize=(16, 12), constrained_layout=True)

        im1 = ax[0].contourf(time, depth, raw_data, levels, cmap='viridis')
        ax[0].set_xlim(xlimits)
        cb1 = fig.colorbar(im1, ax=ax[0])
        cb1.set_ticks([raw_data.min(), raw_data.min() + ((raw_data.max() - raw_data.min()) / 2), raw_data.max()])
        ax[0].set_ylabel('Raw')

        im2 = ax[1].contourf(time, depth, data, levels, cmap ='viridis')
        ax[1].set_xlim(xlimits)
        cb2 = fig.colorbar(im2, ax=ax[1])
        cb2.set_ticks([data.min(), data.min() + ((data.max() - data.min()) / 2), data.max()])
        ax[1].set_ylabel('Filtered')

        ax[2].plot(time, raw_data[np.searchsorted(depth, depthLine), :], "-r", label="Raw")
        ax[2].plot(time, data[np.searchsorted(depth, depthLine), :], "-b", label="Filt.")
        ax[2].set_xlim(xlimits)
        ax[2].set_xlabel('Time (s)')
        ax[2].legend()

        ax[3].plot(depth[start_id_depth:], avg_velo[start_id_depth:], label="Mean vz")
        ax[3].set_xlabel("Depth (mm)")
        ax[3].legend()

        return fig
    
    def save_datafile(self, filename, time, depth, data):
        """
        Saves the corrected UDV data in a file
        """
        data = data.T
        data_to_write = np.zeros(shape = (len(time)+1, len(depth)+1))
        data_to_write[1:,0] = time
        data_to_write[0,1:] = depth
        for i in range(1, data.shape[0]):
            data_to_write[i,1:] = data[i,:]
    
        np.savetxt(open(filename,"w"), data_to_write)