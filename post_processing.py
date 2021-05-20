import pandas as pd
import os
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')

    return y[int(window_len/2-1):-int(window_len/2)]

###################################
offset_frame = 20
result_path = r'result/plot'
processed_result_path = r'result/plot_processed'
os.makedirs(processed_result_path, exist_ok=True)

###################################
angles_info_path = os.path.join(result_path, 'Angles.xlsx')
distances_info_path = os.path.join(result_path, 'Distances.xlsx')

df_angles = pd.read_excel(angles_info_path, sheet_name='Angles')
df_distances = pd.read_excel(distances_info_path, sheet_name='Distances')

df_angles = df_angles.query(f'Frame > {offset_frame}')
df_distances = df_distances.query(f'Frame > {offset_frame}')

## Remove and modify outliers
z_threshlod = 3
for column in df_angles.columns:
    z = np.abs(stats.zscore(df_angles[column]))
    df_angles[column] = np.where(z < z_threshlod, df_angles[column], np.NAN)
    # df_angles[column] = df_angles[column].rolling(window=10, center=True).mean().fillna(method='bfill').fillna(method='ffill')
df_angles.fillna(method='ffill', inplace=True)

for column in df_distances.columns:
    z = np.abs(stats.zscore(df_distances[column]))
    df_distances[column] = np.where(z < z_threshlod, df_distances[column], np.NAN)
    # df_distances[column] = df_distances[column].rolling(window=10, center=True).mean().fillna(method='bfill').fillna(method='ffill')
df_distances.fillna(method='ffill', inplace=True)

## Smoothing
frames = list(df_angles['Frame'])
all_angles = {'Frame': frames}
min_angles = {'Frame': frames}
max_angles = {'Frame': frames}
for column in df_angles.columns:
    if column != 'Frame':
        angles = list(df_angles[column])
        smoothed_angles = savgol_filter(angles, window_length=51, polyorder=3)
        # smoothed_angles = smooth(np.array(angles), window_len=20, window='hanning')
        smoothed_angles = [int(angle) for angle in smoothed_angles]
        all_angles[column] = smoothed_angles
        for i, angle in enumerate(smoothed_angles):
            if column in min_angles.keys():
                previous_min_angle = min_angles[column][i-1]
                if angle < previous_min_angle:
                    min_angles[column].append(angle)
                else:
                    min_angles[column].append(previous_min_angle)
            else:
                min_angles[column] = [angle]
            ##
            if column in max_angles.keys():
                previous_max_angle = max_angles[column][i-1]
                if angle > previous_max_angle:
                    max_angles[column].append(angle)
                else:
                    max_angles[column].append(previous_max_angle)
            else:
                max_angles[column] = [angle]


frames = list(df_distances['Frame'])
all_distances = {'Frame': frames}
min_distances = {'Frame': frames}
max_distances = {'Frame': frames}
for column in df_distances.columns:
    if column != 'Frame':
        distances = list(df_distances[column])
        smoothed_distances = savgol_filter(distances, window_length=51, polyorder=3)
        # smoothed_distances = smooth(smoothed_distances, window_len=10, window='hanning')
        # smoothed_distances = smooth(np.array(distances), window_len=20, window='hanning')
        smoothed_distances = [round(distance, 2) for distance in smoothed_distances]
        all_distances[column] = smoothed_distances
        for i, distance in enumerate(smoothed_distances):
            if column in min_distances.keys():
                previous_min_distance = min_distances[column][i-1]
                if distance < previous_min_distance:
                    min_distances[column].append(distance)
                else:
                    min_distances[column].append(previous_min_distance)
            else:
                min_distances[column] = [distance]
            ##
            if column in max_distances.keys():
                previous_max_distance = max_distances[column][i-1]
                if distance > previous_max_distance:
                    max_distances[column].append(distance)
                else:
                    max_distances[column].append(previous_max_distance)
            else:
                max_distances[column] = [distance]


processed_angles_info_path = os.path.join(processed_result_path, 'Angles.xlsx')
df_angles = pd.DataFrame.from_dict(all_angles)
df_min_angles = pd.DataFrame.from_dict(min_angles)
df_max_angles = pd.DataFrame.from_dict(max_angles)
with pd.ExcelWriter(processed_angles_info_path) as writer:
    df_angles.to_excel(writer, sheet_name='Angles', index=False)
    df_min_angles.to_excel(writer, sheet_name='Min_Angles', index=False)
    df_max_angles.to_excel(writer, sheet_name='Max_Angles', index=False)

processed_distances_info_path = os.path.join(processed_result_path, 'Distances.xlsx')
df_distances = pd.DataFrame.from_dict(all_distances)
df_min_distances = pd.DataFrame.from_dict(min_distances)
df_max_distances = pd.DataFrame.from_dict(max_distances)
with pd.ExcelWriter(processed_distances_info_path) as writer:
    df_distances.to_excel(writer, sheet_name='Distances', index=False)
    df_min_distances.to_excel(writer, sheet_name='Min_Distances', index=False)
    df_max_distances.to_excel(writer, sheet_name='Max_Distances', index=False)

## Plot
for column in df_angles.columns:
    if column != 'Frame':
        angles = list(df_angles[column])
        min_angles = list(df_min_angles[column])
        max_angles = list(df_max_angles[column])
        ##
        plt.figure()
        plt.plot(frames, angles)
        plt.plot(frames, min_angles, linestyle='--', dashes=(5, 3))
        plt.plot(frames, max_angles, linestyle='--', dashes=(5, 3))
        plt.xlabel('Frames')
        plt.ylabel('Angle (degree)')
        plt.title(column)
        plt.grid(True)
        plt.savefig(os.path.join(processed_result_path, column + ".jpg"))
        plt.close()

for column in df_distances.columns:
    if column != 'Frame':
        distances = list(df_distances[column])
        min_distances = list(df_min_distances[column])
        max_distances = list(df_max_distances[column])
        ##
        plt.figure()
        plt.plot(frames, distances)
        plt.plot(frames, min_distances, linestyle='--', dashes=(5, 3))
        plt.plot(frames, max_distances, linestyle='--', dashes=(5, 3))
        plt.xlabel('Frames')
        plt.ylabel('Distance (cm)')
        plt.title(column)
        plt.grid(True)
        plt.savefig(os.path.join(processed_result_path, column + ".jpg"))
        plt.close()