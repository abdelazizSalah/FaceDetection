# @author Abdelaziz Salah
# @date 29/12/2022
# @brief This file contains the implementation of the rectangle region class
# which is used to get the sum of pixels in a specific region instead of iterating every
# time over all the region, we just compute it with certain equation
# so this reduces the time complexity of the algorithm from O(n) to O(1).


class RectangleRegion:
    def __init__(self, startPosX, startPosY, width, height):
        # the x coordinate of the left top corner of the region
        self.startPosX = startPosX

        # the y coordinate of the left top corner of the region
        self.startPosY = startPosY

        # the width of the region
        self.width = width

        # the height of the region
        self.height = height

    # this function is used to get the sum of pixels in the region
    # @param integralImg: the integral image of the image
    # @param scale: the scale of the image by which we need to increase the window size.
    def get_region_sum(self, integralImg, scale=1.0):
        # defining the 4 indcies of the region
        top_left_x = int(self.startPosX * scale)
        top_left_y = int(self.startPosY * scale)
        bottom_right_x = int(self.startPosX * scale) + \
            int(self.width * scale) - 1  # -1 to work with 0 based
        bottom_right_y = int(self.startPosY * scale) + \
            int(self.height * scale) - 1

        # the sum of pixels in the region
        sum_of_regios = int(integralImg[bottom_right_x, bottom_right_y])  # D

        # applying the equation of the rectangle
        # sum = A + D - C - B

        # C
        if top_left_x > 0:
            sum_of_regios -= int(integralImg[top_left_x - 1, bottom_right_y])

        # B
        if top_left_y > 0:
            sum_of_regios -= int(integralImg[bottom_right_x, top_left_y - 1])

        # A
        if top_left_x > 0 and top_left_y > 0:
            sum_of_regios += int(integralImg[top_left_x - 1, top_left_y - 1])

        return sum_of_regios
