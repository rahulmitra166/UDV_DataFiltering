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
    
    def plot_data(self, fig_title, fig_num, time, depth, data, xlimits, levels=300):
        fig = plt.figure(fig_num)
        plt.clf()
        plt.gcf().set_size_inches([16,4])

        plt.contourf(time, depth, data, levels, cmap='viridis')
        plt.title(fig_title)
        plt.xlim(xlimits)
        cb = plt.colorbar()
        cb.set_ticks([data.min(), data.min()+((data.max()-data.min())/2), data.max()])
        plt.ylabel('Depth (mm)')
        plt.xlabel("Time (s)")
        plt.tight_layout()

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