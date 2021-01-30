# Julia Suter, 2020
# Project: Employing the Scene Graph for Phrase Grounding
# Master Thesis, University of Heidelberg

# box_geometry.py
# ----------------
# Functions for computing bounding box size, overlap between two boxes
# and compute union boxes.


def iou(boxA, boxB):
    """Compute the intersection over union for two unite_boxes
    given their coordinates  (xmin, ymin, xmax, ymax)."""

    # Load data
    xmin = (boxA[0], boxB[0])
    ymin = (boxA[1], boxB[1])
    xmax = (boxA[2], boxB[2])
    ymax = (boxA[3], boxB[3])

    # Get min and max coordiantes
    max_ymax = max(ymax)
    min_ymax = min(ymax)
    max_ymin = max(ymin)
    min_ymin = min(ymin)

    max_xmax = max(xmax)
    min_xmax = min(xmax)
    max_xmin = max(xmin)
    min_xmin = min(xmin)

    # Handle edge cases (no overlap at all)
    if xmin[0] > xmax[1] or xmax[0] < xmin[1]:
        return 0

    if ymax[0] < ymin[1] or ymin[0] > ymax[1]:
        return 0

    # Compute union of both boxes
    union = (max(max_ymax, min_ymin) - min(max_ymax, min_ymin)) * (max(max_xmax, min_xmin) - min(max_xmax, min_xmin))

    # Compute intersection of both boxes
    intersection = (max(min_ymax, max_ymin) - min(min_ymax, max_ymin)) * (max(min_xmax, max_xmin) - min(min_xmax, max_xmin))

    # Compute the area for both boxes
    A_1 = (ymax[0] - ymin[0]) * (xmax[0]-xmin[0])
    A_2 = (ymax[1] - ymin[1]) * (xmax[1]-xmin[1])

    # Compute the union (both areas minus intersection)
    union = A_1 + A_2 - intersection

    # Return IoU
    return intersection/union


def overlap_iou(coords_1, coords_2, threshold=0.5):
    """Check whether intersection over union (IoU) >= 0.5
    given two sets of coordinates (xmin, ymin, xmax, ymax).
    Used to check whether two boxes match."""

    # Get IoU
    iou_score = iou(coords_1, coords_2)

    # Does it surpass threshold?
    if iou_score >= threshold:
        return True
    else:
        return False


def box_union(boxA, boxB):
    """Get the union coordinates of two boxes
    given their coordinates (xmin, ymin, xmax, ymax)."""

    # Get the min and max points
    xA = min(boxA[0], boxB[0])
    yA = min(boxA[1], boxB[1])
    xB = max(boxA[2], boxB[2])
    yB = max(boxA[3], boxB[3])

    # Return coordinates
    return xA, yA, xB, yB


def unite_boxes(box_list):
    """Unite several boxes given their coordinates (xmin, ymin, xmax, ymax)
    and return coordinates."""

    union = None

    # For each box
    for box in box_list:

        # Use first box as default
        if union is None:
            union = box

        # Combine with current union box
        else:
            union = box_union(union, box)

    # Return union box coordinates
    return union


def get_box_size(coordinates):
    """Compute box size given coordinates (xmin, ymin, xmax, ymax)."""

    # Get coordinates
    xmin, ymin, xmax, ymax = coordinates

    # Get width and height
    width = xmax-xmin
    height = ymax-ymin

    # Compute size
    size = width*height

    return size


def iou_deprecated(boxA, boxB):
    """
    CAVEAT: this code is buggy because of the +1 fix. Not a clean implementation
    and thus not used in our project!
    Determine the (x, y)-coordinates of the intersection rectangle.
    Source: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/"""

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou
