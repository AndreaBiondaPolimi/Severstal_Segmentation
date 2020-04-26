import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def show_imgs (imgs, titles, messages):
    n_img = len(imgs)
    _ , axarr = plt.subplots(n_img)
    
    for i in range (n_img):
        if i==0:
            axarr[i].imshow(imgs[i])
        else:
            axarr[i].imshow(imgs[i])
        axarr[i].set_title(titles[i])
        axarr[i].text(0, 200, messages[i])
        axarr[i].axis('off')

    fig = plt.gcf()
    fig.set_size_inches(14, 8, forward=True)
    plt.tight_layout()

    plt.show()


def restructure_data_frame(path):
    train = pd.read_csv(path)

    # RESTRUCTURE TRAIN DATAFRAME
    train['ImageId'] = train['ImageId_ClassId'].map(lambda x: x.split('.')[0]+'.jpg')

    train2 = pd.DataFrame({'ImageId':train['ImageId'][::4]})
    train2['e1'] = train['EncodedPixels'][::4].values
    train2['e2'] = train['EncodedPixels'][1::4].values
    train2['e3'] = train['EncodedPixels'][2::4].values
    train2['e4'] = train['EncodedPixels'][3::4].values
    train2.reset_index(inplace=True,drop=True)
    train2.fillna('',inplace=True)
    train2['count'] = np.sum(train2.iloc[:,1:]!='',axis=1).values
    train2.head()

    return train2