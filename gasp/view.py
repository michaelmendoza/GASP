import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import seaborn as sns

def view3D(filename: str, data3D: np.ndarray, path: str='./images/') -> None:

    # set seaborn darkgrid theme
    sns.set_theme(style="darkgrid")

    fig, ax = plt.subplots()

    def animate(frame_num):
        ax.clear()
        ax.imshow(abs(data3D[:, :, frame_num]), cmap='gray')
        return ax

    anim = FuncAnimation(fig, animate, frames=data3D.shape[2], interval=1)

    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

    path = os.path.join(path, filename)
    print(path)
    anim.save(path)
