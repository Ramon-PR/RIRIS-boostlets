import numpy as np
import matplotlib.pyplot as plt

def plot_array_images(array, num_cols=5, cmap='gray'):

    if type(array) == list:
        array = np.stack(array, axis=2)

    if array.dtype == "complex128":
        array = np.abs(array)
        
    num_rows = int(np.ceil(array.shape[2] / num_cols))
        
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 3))

    for i in range(array.shape[2]):
        row = i // num_cols
        col = i % num_cols

        ax = axs[row, col]
        c = ax.pcolor(array[:, :, i], cmap=cmap)
        cbar = fig.colorbar(c, ax=ax, orientation='vertical', format='%1.1f')
        ax.set_title(f'Index {i}')
        ax.axis('off')


    # Ocultar cualquier subtrama vacía
    if num_rows * num_cols > array.shape[2]:
        for j in range(array.shape[2], num_rows * num_cols):
            fig.delaxes(axs.flatten()[j])

    plt.tight_layout()
    plt.show()


def plot_ls_images(images, titles):

    images = [np.abs(im) if im.dtype == "complex128" else im for im in images]

    fig, ax = plt.subplots(1, len(images), figsize=(18, 6))
    for i in range(len(images)):
        ax[i].pcolor(images[i])
        ax[i].set_title(titles[i])
        ax[i].axis('off')
    plt.tight_layout()
    plt.show()