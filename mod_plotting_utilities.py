import numpy as np
import matplotlib.pyplot as plt

def plot_array_images(array, num_cols=5):

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
        # ax.imshow(array[:, :, i], cmap='gray')
        # c = ax.pcolor(array[:, :, i], cmap='gray')
        c = ax.pcolor(array[:, :, i], cmap='jet')
        cbar = fig.colorbar(c, ax=ax, orientation='vertical', format='%1.1f')
        # cbar.ax.set_yticklabels(['< -1', '0', '> 1'])  # vertically oriented colorbar

        ax.set_title(f'Index {i}')
        ax.axis('off')


    # Ocultar cualquier subtrama vacía
    if num_rows * num_cols > array.shape[2]:
        for j in range(array.shape[2], num_rows * num_cols):
            fig.delaxes(axs.flatten()[j])

    plt.tight_layout()
    plt.show()