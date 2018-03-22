# Region selection
vertices_ratio                          = [(0, 1), (1/10., 1/2.), (9/10., 1/2.), (1, 1)]

# Gaussin blur kernel size
guassin_blur_kernel_size                = 7

# Color range selection
red_threshold                           = 200
green_threshold                         = 200
blue_threshold                          = 200
rgb_threshold                           = [red_threshold, green_threshold, blue_threshold]

lower_yellow_white                      = [192, 192, 32]
upper_yellow_white                      = [255, 255, 255]

# Canny edgy detection
canny_low_threshold                     = 10
canny_high_threshold                    = 30

# Hough transform
hough_rho                               = 2  # distance resolution in pixels of the Hough grid
hough_theta_scale                       = 180  # angular resolution in radians of the Hough grid
hough_threshold                         = 40  # minimum number of votes (intersections in Hough grid celll)
hough_min_line_length                   = 75  # minimum number of pixels making up a line
hough_max_line_gap                      = 50  # maximum gap in pixels between connectable line segments

# Post processing
# Slope for deciding left or right lane
slope_left_low                          = -1.0
slope_left_high                         = -0.5
slope_right_low                         = 0.5
slope_right_high                        = 1.0

