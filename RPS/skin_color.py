from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
# from skimage.color import rgb2lab, deltaE_cie76
import os
import colorsys





def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def get_color(image, number_of_colors, show_chart=True):
    modified_image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)

    clf = KMeans(n_clusters = number_of_colors)
    labels = clf.fit_predict(modified_image)

    counts = Counter(labels)

    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    if (show_chart):
        fig = plt.figure()
        fig.add_subplot(122)
        plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)

        fig.add_subplot(121)
        plt.imshow(image)

        plt.show(block=False)
        x = input('Enter quit: ')

    return rgb_colors

def convert_rgb_to_hsv(rgb):
    r, g, b = tuple(list(rgb[0]))
    hsv = colorsys.rgb_to_hsv(r/255., g/255., b/255.)
    hsv = [color * 255 for color in hsv]
    return hsv

def rgb_to_hsv(rgb):
    rgb = list(rgb[0])
    r, g, b = rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df/mx)*100
    v = mx*100
    return [h, s, v]



def get_hsv_range(hsv, buffer):
    hsv_range = [[],[]]
    for color in hsv:
        l_val = color - buffer if color - buffer > 0 else 0
        u_val = color + buffer if color + buffer < 255  else 255
        hsv_range[0].append(l_val)
        hsv_range[1].append(u_val)
    return hsv_range

def get_trackbar_values(range_filter):
    values = []

    for i in ["MIN", "MAX"]:
        for j in range_filter:
            v = cv2.getTrackbarPos("%s_%s" % (j, i), "Trackbars")
            values.append(v)

    return values

if __name__ == '__main__':
    img = get_image('/Users/mandywoo/Documents/rock_paper_scissor_project/test_img.jpg')
    # rgb = get_color(img, 1)
    # print(rgb)
    # r, g, b = tuple(list(rgb[0]))
    # print(colorsys.rgb_to_hsv(r, g, b))
    print(get_trackbar_values('HSV'))

# TODO: create a box, get image, get color, get hsv, set hsv limits, detect only those colors, convert to black and white