import cv2
import numpy as np


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(cf, img, lines, color=[255, 0, 0], thickness=6):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    :type cf: object
    """

    def accumulate_line(lane, slope, x1, x2, y1, y2):
        lane['num'] += 1
        lane['slope'] += slope
        lane['x1'] += x1
        lane['y1'] += y1
        lane['x2'] += x2
        lane['y2'] += y2
        return lane

    def average_lines(lane):
        lane['slope'] = lane['slope'] / lane['num']
        lane['x1'] = int(lane['x1'] / lane['num'])
        lane['y1'] = int(lane['y1'] / lane['num'])
        lane['x2'] = int(lane['x2'] / lane['num'])
        lane['y2'] = int(lane['y2'] / lane['num'])
        return lane

    def draw_line(img, lane, yi, thickness):
        ry1 = yi + thickness * 3
        rx1 = int(lane['x2'] + (ry1 - lane['y2']) / lane['slope'])
        ry2 = ysize - 1
        rx2 = int(rx1 + (ry2 - ry1) / lane['slope'])
        cv2.line(img, (rx1, ry1), (rx2, ry2), color, thickness)

    ysize = img.shape[0]
    try:
        # rightline and leftline cumlators
        rl = {'num': 0, 'slope': 0.0, 'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0}
        ll = {'num': 0, 'slope': 0.0, 'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0}
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1)
                if slope > cf.slope_right_low and slope < cf.slope_right_high:
                    # this is a right lane
                    rl = accumulate_line(rl, slope, x1, x2, y1, y2)
                elif slope > cf.slope_left_low and slope < cf.slope_left_high:
                    ll = accumulate_line(ll, slope, x1, x2, y1, y2)

        # average/extrapolate all of the lines that makes the lane
        if rl['num'] > 0 or ll['num'] > 0:
            rl = average_lines(rl)
            ll = average_lines(ll)
            # fint ehr right and left line's intersection
            xi = int(
                (ll['y2'] - rl['y2'] + rl['slope'] * rl['x2'] - ll['slope'] * ll['x2']) / (rl['slope'] - ll['slope']))
            yi = int(rl['y2'] + rl['slope'] * (xi - rl['x2']))
            draw_line(img, rl, yi, thickness)
            draw_line(img, ll, yi, thickness)
    except:
        return -1000, 0.0, 0.0, 0.0, 0, 0
