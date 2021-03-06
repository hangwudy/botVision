import numpy as np
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon


def create_image_annotation(file_name, height, width, image_id):
    images = {
        'license': 1,
        'file_name': file_name,
        'height': height,
        'width': width,
        'id': image_id
    }
    return images






def create_sub_mask_annotation(sub_mask, is_crowd, image_id, category_id, annotation_id):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')
    # print(contours)

    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        
        poly = poly.simplify(0.4, preserve_topology=False)

        # print(poly)
        polygons.append(poly)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentations.append(segmentation)

    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)
    x, y, max_x, max_y = multi_poly.bounds
    width = max_x - x
    height = max_y - y
    bbox = (x, y, width, height)
    area = multi_poly.area

    annotation = {
        'segmentation': segmentations,
        'area': area,
        'iscrowd': is_crowd,
        'image_id': image_id,
        'bbox': bbox,
        'category_id': category_id,
        'id': annotation_id
    }

    return annotation
