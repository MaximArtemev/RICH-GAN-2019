import torch

import matplotlib.pyplot as plt

DLL_DIM = 5
INPUT_DIM = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dll_columns = ['RichDLLe', 'RichDLLk', 'RichDLLmu', 'RichDLLp', 'RichDLLbt']

def plot_distributions(real, gen, w):
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))

    for particle_type, ax in zip((0, 1, 3, 4), axes.flatten()):
        _, bins, _ = ax.hist(
            real[:, particle_type]
            , bins=100
            , label="real"
            , density=True
            , weights=w
        )

        ax.hist(
            gen[:, particle_type]
            , bins=bins
            , label="gen"
            , alpha=0.5
            , density=True
            , weights=w
        )

        ax.legend()
        ax.set_title(dll_columns[particle_type])

    plt.show()

def plot_three_distributions(real, teacher_gen, student_gen, w):
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))

    for particle_type, ax in zip((0, 1, 3, 4), axes.flatten()):
        _, bins, _ = ax.hist(
            real[:, particle_type]
            , bins=100
            , label="real"
            , density=True
            #, weights=w
        )

        ax.hist(
            teacher_gen[:, particle_type]
            , bins=bins
            , label="teacher"
            , alpha=0.5
            , density=True
            #, weights=w
        )

        ax.hist(
            student_gen[:, particle_type]
            , bins=bins
            , histtype='step'
            , label="student"
            , alpha=0.5
            , density=True
            #, weights=w
        )

        ax.legend()
        ax.set_title(dll_columns[particle_type])

    plt.show()
