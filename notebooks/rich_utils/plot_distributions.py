import matplotlib.pyplot as plt
import numpy as np


def plot_joint_dll_distributions(data, dlls):
    fig, axs = plt.subplots(len(dlls), len(dlls), figsize=(17, 15))

    for j, dll1 in enumerate(dlls):
        for i, dll2 in enumerate(dlls):
            q1_1, q1_99 = data[dll1].quantile(0.03), data[dll1].quantile(0.97)
            q2_1, q2_99 = data[dll2].quantile(0.03), data[dll2].quantile(0.97)

            if i == j:
                arr = data[[dll1]].query(f'{q1_1} < {dll1} < {q1_99}')
                axs[i, i].hist(arr[dll1], 100)
                axs[i, i].set_title(dll1)
            else:
                arr = data[[dll1, dll2]] \
                    .query(f'{q1_1} < {dll1} < {q1_99}') \
                    .query(f'{q2_1} < {dll2} < {q2_99}')

                axs[i, j].hist2d(arr[dll1], arr[dll2], (100, 100, ))
                m1, m2 = max(q1_1, q2_1), min(q1_99, q2_99)
                axs[i, j].plot(np.arange(m1, m2, 1), np.arange(m1, m2, 1), c='red')
                axs[i, j].set_title(dll1 + ' ' + dll2)


def plot_joint_cond_dll_distributions(data, dlls, features):
    fig, axs = plt.subplots(len(dlls) + 1, len(features), figsize=(17, 17))

    for j, f in enumerate(features):
        for i, dll in enumerate([None] + dlls):
            q2_1, q2_99 = data[f].quantile(0.03), data[f].quantile(0.97)
            if i == 0:
                arr = data[[f]].query(f'{q2_1} < {f} < {q2_99}')
                axs[i, j].hist(arr[f], 500)
                axs[i, j].set_title(f)
            else:
                q1_1, q1_99 = data[dll].quantile(0.03), data[dll].quantile(0.97)

                arr = data[[dll, f]] \
                    .query(f'{q1_1} < {dll} < {q1_99}') \
                    .query(f'{q2_1} < {f} < {q2_99}')

                axs[i, j].hist2d(arr[f], arr[dll], (100, 100, ))
                axs[i, j].set_title(f + ' ' + dll)
