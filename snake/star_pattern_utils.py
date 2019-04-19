import numpy as np
from shapely.geometry import LineString, Polygon


def r_even_divide(ind, min_radius, max_radius, num_pts):
    r_delta = (max_radius - min_radius) / num_pts
    r = min_radius + r_delta * ind
    return r


def star_pattern_ind_to_image_ind(ind, center, min_radius, max_radius, num_line, num_point_on_line,
                                  max_dim=None, round_to_int=False, center_jitter=None, angle_jitter=None,
                                  divide_function=r_even_divide):
    """
    Convert start pattern at a specific location to index on the image
    :param ind: List[List]
        list of index on each ray in star pattern
        pass None to consider all points on star pattern
    :param center: [int, int] follow [height, width]
    :param num_line: int
    :param min_radius: float
    :param max_radius: float
    :param num_point_on_line: int
    :param max_dim: [int, int]
        max height, max width
    :param round_to_int: bool
    :return: np.ndarray
    num_line * num_point_on_line 2D array contain the correspond index of the star pattern on the image
    Order: row, column
    """
    if ind is None:
        ind = np.asarray([list(range(num_point_on_line))] * num_line)
    assert len(ind) == num_line
    if center_jitter is None:
        center_jitter = [0, 0]
    if angle_jitter is None:
        angle_jitter = 0

    theta_delta = 2 * np.pi / num_line
    rs = divide_function(ind, min_radius, max_radius, num_point_on_line)
    theta = theta_delta * np.arange(num_line) + angle_jitter
    theta = np.expand_dims(theta, -1)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    rows = (rs * sin_t).flatten()
    cols = (rs * cos_t).flatten()
    output = np.stack([rows, cols], -1)
    if not np.any(np.isnan(center)):
        center = (np.asarray(center) + np.asarray(center_jitter)).reshape([1, 2])
        output = output + center
        # output = np.clip(output, 0, np.inf)
        if max_dim:
            output[:, 0] = np.clip(output[:, 0], 0, max_dim[0])
            output[:, 1] = np.clip(output[:, 1], 0, max_dim[1])
        if round_to_int:
            output = np.round(output)
    else:
        output = np.zeros_like(output)

    # import matplotlib.pyplot as plt
    # img = np.zeros((212, 212))
    # plt.imshow(img)
    # for j, pt in enumerate(output):
    #     if j<num_point_on_line:
    #         plt.plot(pt[1], pt[0], 'y.')
    #     else:
    #         plt.plot(pt[1], pt[0], 'g.')
    # plt.show()
    return output


def star_pattern_to_segments(center, radius, num_lines, center_jitter=None, angle_jitter=None):
    # segment follow the row, column format
    ind = np.asarray([[0, radius]] * num_lines)
    output = star_pattern_ind_to_image_ind(ind, center, 1, radius, num_lines, radius,
                                           center_jitter=center_jitter, angle_jitter=angle_jitter)
    output = output.reshape((num_lines, 2, 2))
    return output


def find_intersection(segments, contour, min_radius, max_radius):
    """
    follow row, col format
    return normalized position of intersections on each ray of the star pattern
    normalized position = range from 0 -> 1 denote beginning of ray to end of ray
    """
    #
    # import matplotlib.pyplot as plt
    #
    # plt.axis([0, 212, 212, 0])
    # plt.plot(contour[:, 1], contour[:, 0], 'r')
    # for j in range(len(segments)):
    #     if j == 0:
    #         color = 'r'
    #     elif j == 16:
    #         color = 'black'
    #     else:
    #         color = 'b'
    #     plt.plot(segments[j][:, 1], segments[j][:, 0], color)
    # plt.show()

    contour_poly = Polygon(contour)
    output = []
    for j, s in enumerate(segments):
        s_line = LineString(s)
        try:
            # this check is to prevent bugs sometime in shapely where there is clearly intersection but shapely cannot find
            # in those cases, use index of the previous line as approximation
            intersections = contour_poly.intersection(s_line)
            ray_start, ray_end = np.asarray(intersections.coords)
            # plt.plot(ray_end[1], ray_end[0], marker='o')
            d_is = np.sqrt(((ray_end - ray_start) ** 2).sum())
            d_es = (max_radius - min_radius)
            normalized_pos = d_is / d_es
            output.append([normalized_pos])
        except:
            output.append(None)
    # find the last available index
    j = len(output) - 1
    while output[j] is None:
        j = j - 1
    last_index = output[j]
    # do a for loop to correct invalid indices while updating last available index at the same time
    for j in range(len(output)):
        if output[j] is None:
            output[j] = last_index
        else:
            last_index = output[j]
    # plt.show()
    assert len(output) == len(segments)
    output = np.asarray(output)
    return output


if __name__ == '__main__':
    from skimage.io import imread
    from skimage import measure
    import matplotlib.pyplot as plt
    from matplotlib import collections as mc

    label_im = imread("axial_CT_slice_label.bmp").astype(np.float32) / 255

    contour = measure.find_contours(label_im, 0.8)[0]
    # row: 140, column: 160
    c_center = 140, 160
    c_min_radius = 1
    c_max_radius = 60
    num_line = 20
    num_point_on_line = 20
    c_center_offset = [10, 10]
    angle_offset = 0.0
    segment_points = star_pattern_ind_to_image_ind(
        None, c_center, c_min_radius, c_max_radius, num_line, num_point_on_line,
        center_jitter=c_center_offset, angle_jitter=angle_offset)

    fig, ax = plt.subplots()
    ax.imshow(label_im, interpolation='nearest', cmap=plt.cm.gray)
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')

    ax.scatter(segment_points[:, 1], segment_points[:, 0], color='m', marker='.')

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
