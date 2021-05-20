import numpy as np
import matplotlib.pyplot as plt

import itertools
import pandas as pd
import os
from numpy import polyfit
from sklearn.mixture import GaussianMixture as GMM

# of course this is a fake one just to offer an example
def source():
    return itertools.cycle((1, 0, 1, 4, 8, 2, 1, 3, 3, 2))


# import pylab as plt
import scipy as sp

# Generate some test data, i.e. our "observations" of the signal
L = 130
vals = source()

X = []
for i,x in enumerate(vals):
    X.append(x)
    if i>L:
        break


#########################
processed_result_path = r'result/plot_processed_t2'
distances_info_path = os.path.join(processed_result_path, 'Distances.xlsx')
df_distances = pd.read_excel(distances_info_path, sheet_name='Distances')
X = df_distances['Distance_5']
X = np.array(X)

############################
n_components = np.arange(1, 10)
models = [GMM(n, covariance_type='full', random_state=0).fit(X.reshape(-1, 1)) for n in n_components]

plt.plot(n_components, [m.bic(X.reshape(-1, 1)) for m in models], label='BIC')
plt.plot(n_components, [m.aic(X.reshape(-1, 1)) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components')
plt.show()


