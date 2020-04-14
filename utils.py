import matplotlib.pyplot as plt
import numpy as np


def show_imgs (imgs, titles, messages):
    n_img = len(imgs)
    _ , axarr = plt.subplots(n_img)
    
    for i in range (n_img):
        if i==0:
            axarr[i].imshow(np.reshape(imgs[i],(128,800,3)))
        else:
            axarr[i].imshow(np.reshape(imgs[i],(128,800)))
        axarr[i].set_title(titles[i])
        axarr[i].text(0, 200, messages[i])
        axarr[i].axis('off')

    fig = plt.gcf()
    fig.set_size_inches(14, 8, forward=True)
    plt.tight_layout()

    plt.show()

