import base64
import datetime
import io
import json
import os
import time
import webbrowser

import colorama
import cv2
import numpy

from PIL import Image

# ensure keras using tensorflow backend
os.environ["KERAS_BACKEND"] = "tensorflow"

# import modules from keras
from keras.models import model_from_json

try:
    with open("fannet/models/fannet.json", "r") as fp:
        NET_F = model_from_json(fp.read())
    with open("colornet/models/colornet.json", "r") as fp:
        NET_C = model_from_json(fp.read())

    NET_F.load_weights("fannet/checkpoints/fannet_weights.h5")
    NET_C.load_weights("colornet/checkpoints/colornet_weights.h5")
except:
    RUN_FLAG = False

# get opencv version number
def opencv_version():
    return int(cv2.__version__.split(".")[0])


# -----------------------------------------------------------------------------


def select_region(event, x, y, flags, points):
    # no reference to points list is passed
    if points is None or not type(points) is list:
        return

    # handle events
    # - insert : left click
    # - remove : right click
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) > 4:
            points.clear()
        elif len(points) == 4:
            points.pop(0)
        points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(points) > 0:
            points.pop()
    return


# -----------------------------------------------------------------------------


# sort points in top-left, top-right, bottom-right, bottom-left order
def sort_points(points):
    points = sorted(points, key=lambda x: x[1])
    points = sorted(points[:2], key=lambda x: x[0]) + sorted(
        points[2:], key=lambda x: x[0], reverse=True
    )
    return points


# -----------------------------------------------------------------------------


# scale points
def scale_points(points, fscale=1.0):
    scaled_points = points.copy()
    for point in scaled_points:
        point[0] = round(point[0] * fscale)
        point[1] = round(point[1] * fscale)
    return scaled_points


# -----------------------------------------------------------------------------


# draw region
def draw_region(image, points):
    output = image.copy()
    points = sort_points(points)
    npoint = len(points)
    for i in range(npoint):
        cv2.line(
            output, points[i], points[(i + 1) % npoint], (0, 0, 255), 1, cv2.LINE_AA
        )
    for i in range(npoint):
        cv2.circle(output, points[i], 5, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(output, points[i], 4, (0, 255, 0), -1, cv2.LINE_AA)
    return output


# -----------------------------------------------------------------------------


# binarize image
def binarize(image, points=None, thresh=128, maxval=255, thresh_type=0):
    # convert image to grayscale
    image = image.copy()

    # remove everything except the region bounded by given points
    if not points is None and type(points) is list and len(points) > 2:
        points = sort_points(points)
        points = numpy.array(points, numpy.int64)
        mask = numpy.zeros_like(image, numpy.uint8)
        cv2.fillConvexPoly(mask, points, (255, 255, 255), cv2.LINE_AA)
        image = cv2.bitwise_and(image, mask)

    # estimate mask 1 from MSER
    msers = cv2.MSER_create().detectRegions(image)[0]
    setyx = set()
    for region in msers:
        for point in region:
            setyx.add((point[1], point[0]))
    setyx = tuple(numpy.transpose(list(setyx)))
    mask1 = numpy.zeros(image.shape, numpy.uint8)
    mask1[setyx] = maxval

    # estimate mask 2 from thresholding
    mask2 = cv2.threshold(image, thresh, maxval, thresh_type)[1]

    # get binary image from estimated masks
    image = cv2.bitwise_and(mask1, mask2)

    return image


# -----------------------------------------------------------------------------


# find contours
def find_contours(image, min_area=0, sort=True):
    # convert image to grayscale
    image = image.copy()

    # find contours
    if opencv_version() == 3:
        contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
    else:
        contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

    # filter contours by area
    contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area]

    if len(contours) < 1:
        return ([], [])

    # sort contours from left to right using respective bounding boxes
    if sort:
        bndboxes = [cv2.boundingRect(contour) for contour in contours]
        contours, bndboxes = zip(
            *sorted(zip(contours, bndboxes), key=lambda x: x[1][0])
        )

    return contours, bndboxes


# -----------------------------------------------------------------------------


# draw contours
def draw_contours(image, contours, index, color=(0, 255, 0), color_mode=None):
    image = cv2.cvtColor(image, color_mode) if color_mode else image.copy()
    drawn = numpy.zeros_like(image, numpy.uint8)
    for i in range(len(contours)):
        drawn = cv2.drawContours(drawn, contours, i, (255, 255, 255), -1, cv2.LINE_AA)
    if len(contours) > 0 and index >= 0:
        drawn = cv2.drawContours(drawn, contours, index, color, -1, cv2.LINE_AA)
    image = cv2.bitwise_and(drawn, image)
    return image


# -----------------------------------------------------------------------------


# grab region of interest
def grab_region(image, bwmask, contours, bndboxes, index):
    region = numpy.zeros_like(bwmask, numpy.uint8)
    if len(contours) > 0 and len(bndboxes) > 0 and index >= 0:
        x, y, w, h = bndboxes[index]
        region = cv2.drawContours(
            region, contours, index, (255, 255, 255), -1, cv2.LINE_AA
        )
        region = region[y : y + h, x : x + w]
        bwmask = bwmask[y : y + h, x : x + w]
        bwmask = cv2.bitwise_and(region, region, mask=bwmask)
        region = image[y : y + h, x : x + w]
        region = cv2.bitwise_and(region, region, mask=bwmask)
    return region


# -----------------------------------------------------------------------------


# grab all regions of interest
def grab_regions(image, image_mask, contours, bndboxes):
    regions = []
    for index in range(len(bndboxes)):
        regions.append(grab_region(image, image_mask, contours, bndboxes, index))
    return regions


# -----------------------------------------------------------------------------


# convert image to tensor
def image2tensor(image, shape, padding=0.0, rescale=1.0, color_mode=None):
    output = cv2.cvtColor(image, color_mode) if color_mode else image.copy()
    output = numpy.atleast_3d(output)
    rect_w = output.shape[1]
    rect_h = output.shape[0]
    sqrlen = int(numpy.ceil((1.0 + padding) * max(rect_w, rect_h)))
    sqrbox = numpy.zeros((sqrlen, sqrlen, output.shape[2]), numpy.uint8)
    rect_x = (sqrlen - rect_w) // 2
    rect_y = (sqrlen - rect_h) // 2
    sqrbox[rect_y : rect_y + rect_h, rect_x : rect_x + rect_w] = output
    output = cv2.resize(sqrbox, shape[:2])
    output = numpy.atleast_3d(output)
    output = numpy.asarray(output, numpy.float32) * rescale
    output = output.reshape((1,) + output.shape)
    return output


# -----------------------------------------------------------------------------


# convert character to one-hot encoding
def char2onehot(character, alphabet):
    onehot = [0.0] * len(alphabet)
    onehot[alphabet.index(character)] = 1.0
    onehot = numpy.asarray(onehot, numpy.float32).reshape(1, len(alphabet), 1)
    return onehot


# -----------------------------------------------------------------------------


# resize image
def resize(image, w=-1, h=-1, bbox=False):
    image = Image.fromarray(image)
    bnbox = image.getbbox() if bbox else None
    image = image.crop(bnbox) if bnbox else image
    if w <= 0 and h <= 0:
        w = image.width
        h = image.height
    elif w <= 0 and h > 0:
        w = int(image.width / image.height * h)
    elif w > 0 and h <= 0:
        h = int(image.height / image.width * w)
    else:
        pass
    image = image.resize((w, h))
    image = numpy.asarray(image, numpy.uint8)
    return image


# -----------------------------------------------------------------------------


# update bounding boxes
def update_bndboxes(bndboxes, index, image):
    change_x = (image.shape[1] - bndboxes[index][2]) // 2
    bndboxes = list(bndboxes)
    for i in range(0, index + 1):
        x, y, w, h = bndboxes[i]
        bndboxes[i] = (x - change_x, y, w, h)
    for i in range(index + 1, len(bndboxes)):
        x, y, w, h = bndboxes[i]
        bndboxes[i] = (x + change_x, y, w, h)
    bndboxes = tuple(bndboxes)
    return bndboxes


# -----------------------------------------------------------------------------


# paste images
def paste_images(image, patches, bndboxes):
    image = Image.fromarray(image)
    for patch, bndbox in zip(patches, bndboxes):
        patch = Image.fromarray(patch)
        image.paste(patch, bndbox[:2])
    image = numpy.asarray(image, numpy.uint8)
    return image


# -----------------------------------------------------------------------------


# inpaint image
def inpaint(image, mask):
    k = numpy.ones((5, 5), numpy.uint8)
    m = cv2.dilate(mask, k, iterations=1)
    i = cv2.inpaint(image, m, 10, cv2.INPAINT_TELEA)
    return i


# -----------------------------------------------------------------------------


# transfer color having maximum occurence
def transfer_color_max(source, target):
    colors = source.convert("RGB").getcolors(256 * 256 * 256)
    colors = sorted(colors, key=lambda x: x[0], reverse=True)
    maxcol = (
        colors[0][1]
        if len(colors) == 1
        else colors[0][1] if colors[0][1] != (0, 0, 0) else colors[1][1]
    )
    output = Image.new("RGB", target.size)
    colors = Image.new("RGB", target.size, maxcol)
    output.paste(colors, (0, 0), target.convert("L"))
    return output


# -----------------------------------------------------------------------------


# transfer color using approximate pallet
def transfer_color_pal(source, target):
    source = source.convert("RGB")
    src_bb = source.getbbox()
    src_bb = source.crop(src_bb) if src_bb else source.copy()
    colors = Image.new("RGB", src_bb.size)
    src_np = numpy.asarray(src_bb, numpy.uint8)
    for i in range(src_np.shape[0]):
        row_np = src_np[i].reshape(1, -1, 3)
        col_id = numpy.where(row_np == 0)[1]
        row_np = numpy.delete(row_np, col_id, axis=1)
        row_im = Image.fromarray(row_np).resize((colors.width, 1))
        colors.paste(row_im, (0, i))
    target = target.convert("L")
    colors = colors.resize(target.size)
    output = Image.new("RGB", target.size)
    output.paste(colors, (0, 0), target)
    return output


# -----------------------------------------------------------------------------


# edit character
def edit_char(
    image, image_mask, contours, bndboxes, index, char, alphabet, fannet, colornet
):
    # validate parameters
    if (
        len(contours) <= 0
        or len(bndboxes) <= 0
        or len(contours) != len(bndboxes)
        or index < 0
    ):
        return

    # generate character
    region_f = grab_region(image_mask, image_mask, contours, bndboxes, index)
    tensor_f = image2tensor(region_f, fannet.input_shape[0][1:3], 0.1, 1.0)
    onehot_f = char2onehot(char, alphabet)
    output_f = fannet.predict([tensor_f, onehot_f])
    output_f = numpy.squeeze(output_f)
    output_f = numpy.asarray(output_f, numpy.uint8)

    # transfer color
    region_c = grab_region(image, image_mask, contours, bndboxes, index)
    source_c = Image.fromarray(region_c)
    target_f = Image.fromarray(output_f)
    output_c = transfer_color_max(source_c, target_f)
    output_c = numpy.asarray(output_c, numpy.uint8)

    output_f = resize(output_f, -1, region_f.shape[0], True)
    output_c = resize(output_c, -1, region_c.shape[0], True)

    # inpaint old layout
    mpatches = grab_regions(image_mask, image_mask, contours, bndboxes)
    o_layout = numpy.zeros_like(image_mask, numpy.uint8)
    o_layout = paste_images(o_layout, mpatches, bndboxes)
    inpainted_image = inpaint(image, o_layout)

    # create new layout
    bpatches = grab_regions(image, image_mask, contours, bndboxes)
    bndboxes = update_bndboxes(bndboxes, index, output_f)
    bpatches[index] = output_c
    n_layout = numpy.zeros_like(image, numpy.uint8)
    n_layout = paste_images(n_layout, bpatches, bndboxes)
    mpatches[index] = output_f
    m_layout = numpy.zeros_like(image_mask, numpy.uint8)
    m_layout = paste_images(m_layout, mpatches, bndboxes)

    # generate final result
    n_layout = Image.fromarray(n_layout)
    m_layout = Image.fromarray(m_layout)
    inpainted_image = Image.fromarray(inpainted_image)
    inpainted_image.paste(n_layout, (0, 0), m_layout)

    layout = numpy.asarray(m_layout, numpy.uint8)
    edited = numpy.asarray(inpainted_image, numpy.uint8)

    return (layout, edited)
