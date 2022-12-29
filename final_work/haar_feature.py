# @author Abdlaziz Salah
# @date 29/12/2022
# @brief This file contains the implementation of the haar feature class
# which computes a value of a haar feature in a specific region of the integral image

class HaarFeature:
    # each haar feature is composed of 2 symmetric regions: black and white
    # the positive region is the white region
    # the negative region is the black region
    # and the value of this haar feature is sub of black and white regions.
    def __init__(self, positive_region, negative_region):
        # the positive region of the haar feature which is an object from the rectangle class
        self.positive_region = positive_region

        # the negative region of the haar feature which is an object from the rectangle class
        self.negative_region = negative_region

    # this function is used to get the value of the haar feature in a specific region
    # @param integeralImg: the integral image of the image
    # @param scale: the scale of the image by which we need to increase the window size.
    def get_haar_feature_value(self, integeralImg, scale=1.0):
        positive_sum = sum([rectangle.get_region_sum(integeralImg, scale)
                           for rectangle in self.positive_region])

        negative_sum = sum([rectangle.get_region_sum(integeralImg, scale)
                           for rectangle in self.negative_region])

        # computing the haar value.
        return negative_sum - positive_sum
