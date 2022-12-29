# @author Abdelaziz Salah
# @date 29/12/2022
# @brief this file contians utility functions used for extracting the haar features.

from rectangle_region import RectangleRegion
import numpy as np
# import commonfunctions as cf  # this a custom module found the commonfunctions.py


# this function is used to get the integral image of the image
def integeral_img(img):
    row = len(img.shape[0])
    col = len(img.shape[1])
    integral_img = np.zeros((row, col))
    for r in range(row):
        for c in range(col):
            if r == 0 and c == 0:  # first element so we add only its value
                integral_img[r, c] = img[r, c]
            elif r == 0:  # first row so we add the value of the current element and the previous element
                integral_img[r, c] = integral_img[r, c - 1] + img[r, c]
            elif c == 0:  # first column so we add the value of the current element and the previous element
                integral_img[r, c] = integral_img[r - 1, c] + img[r, c]
            else:  # we add the value of the current element and the previous element in the same row and the previous element in the same column and subtract the element in common from the previous values.
                integral_img[r, c] = integral_img[r - 1, c] + \
                    integral_img[r, c - 1] - \
                    integral_img[r - 1, c - 1] + img[r, c]
            # note (\) is not a division it is because the editor want to make a new line. so it doesn't affect the main equation. we just apply sum and subtractions.

    return integral_img


# this function is used to generate values from the haar features.
# @param img_width, img_height: the width and height of the image.
# @param shift: the amount of distance by which the window will be shifted.
# @param min_width: the minimum width of the haar feature.
# @param min_height: the minimum height of the haar feature.
def build_features(img_width, img_height, shift=1, min_width=1, min_height=1):
    # this will be a tuple of (positive(white regions) , negative(black regions))
    features = []

    # we will loop over the image and generate haar features.
    for window_width in range(min_width, img_width + 1):
        for window_height in range(min_height, img_height + 1):
            top_leftX = 0
            while top_leftX + window_width <= img_width:
                top_leftY = 0
                while top_leftY + window_height <= img_height:
                    # now we need to generate all possible haar regions
                    immediate = RectangleRegion(
                        top_leftX, top_leftY, window_width, window_height)
                    right = RectangleRegion(
                        top_leftX + window_width, top_leftY, window_width, window_height)
                    bottom = RectangleRegion(
                        top_leftX, top_leftY + window_height, window_width, window_height)
                    bottom_right = RectangleRegion(
                        top_leftX + window_width, top_leftY + window_height, window_width, window_height)
                    # for 3 rectangles type.
                    right_3 = RectangleRegion(
                        top_leftX + 2 * window_width, top_leftY, window_width, window_height)
                    bottom_3 = RectangleRegion(
                        top_leftX, top_leftY + 2 * window_height, window_width, window_height)

                    # now we have 3 types of rectangles.
                    # 1- 2 rectangles type. -> vertical and horizontal
                    # 2- 3 rectangles type. -> vertical and horizontal
                    # 3- 4 rectangles type. -> diagonal.

                    # lets append all possible 2 rectangles type.
                    if(top_leftX + 2 * window_width < img_width):
                        features.append((immediate, right))

                    if(top_leftY + 2 * window_height < img_height):
                        features.append((immediate, bottom))

                    # lets append all possible 3 rectangles type.
                    if(top_leftX + 3 * window_width < img_width):
                        features.append((immediate, right, right_3))

                    if(top_leftY + 3 * window_height < img_height):
                        features.append((immediate, bottom, bottom_3))

                    # lets append diagonal if possible.
                    if(top_leftX + 2 * window_width < img_width and top_leftY + 2 * window_height < img_height):
                        features.append((immediate, bottom_right))

                    # after generating all possible haar features we will shift the window by the shift value.
                    top_leftY += shift
                top_leftX += shift

    return features


# this function is used to build all features for all training data.
# @param integral_images: a list of all integral images which are the training data.
# @param features: a list of all haar features.
def apply_feature(integral_images, features):
    # each row will contain a list of features , for example:
    # feature[0][i] is the first feature of the image of index i in the data set..
    # y: will be kept as it is => f0=([...], y); f1=([...], y),...
    ourFeature = np.zeros(
        (len(features), len(integral_images)), dtype=np.int32)

    # we will loop over all images and apply all features on it.
    for i, feature in enumerate(features):
        ourFeature[i] = list(map(lambda integralImg: feature.get_haar_feature_value(
            integralImg), integral_images))

    # now we will return all the features.
    return ourFeature
