import cv2
import numpy as np


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param.append([x, y])


def select_points(img):
    """
    Prompts the user to select two points in the image.
    """
    points = []
    cv2.imshow("Select two points", img)
    cv2.setMouseCallback("Select two points", mouse_callback, points)
    while len(points) < 2:
        cv2.waitKey(1)
    cv2.destroyWindow("Select two points")
    return np.array(points, dtype=np.float32)


def compute_scale_factor(src_points, dst_points, distance):
    # Compute the distance between the points in the perspective view
    src_distance = np.linalg.norm(np.subtract(src_points[0], src_points[1]))

    # Compute the distance between the points in the bird's-eye view
    dst_distance = np.linalg.norm(np.subtract(dst_points[0], dst_points[1]))

    # Compute the scale factor between the perspective and bird's-eye views
    scale_factor = distance / dst_distance

    return scale_factor


# function to select four points on a image to capture desired region
def draw_circle(event, x, y, flags, param):
    global image
    global pointIndex
    global pts

    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        pts[pointIndex] = (x, y)
        # print(pointIndex)
        if pointIndex == 3:
            cv2.line(image, pts[0], pts[1], (0, 255, 0), thickness=2)
            cv2.line(image, pts[0], pts[2], (0, 255, 0), thickness=2)
            cv2.line(image, pts[1], pts[3], (0, 255, 0), thickness=2)
            cv2.line(image, pts[2], pts[3], (0, 255, 0), thickness=2)
        pointIndex = pointIndex + 1


def generate_birds_eye_view(perspective_img):
    # Display the image and allow the user to select four points in the perspective image
    cv2.imshow("Select four points in the perspective image", perspective_img)
    src_points = []
    cv2.setMouseCallback(
        "Select four points in the perspective image", mouse_callback, src_points
    )
    while len(src_points) < 4:
        cv2.waitKey(1)
    cv2.destroyWindow("Select four points in the perspective image")
    src_points = np.array(src_points, dtype=np.float32)

    # Compute the size of the output image
    height, width, _ = perspective_img.shape
    output_size = (width + 3000, height + 3000)

    # Compute the destination points based on the selected points
    min_x, min_y = np.int32(src_points.min(axis=0))
    max_x, max_y = np.int32(src_points.max(axis=0))
    dst_points = np.array(
        [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]],
        dtype=np.float32,
    )

    # Compute the homography using the selected points
    H, _ = cv2.findHomography(src_points, dst_points)

    # Transform the perspective image to the bird's-eye view
    birds_eye_view = cv2.warpPerspective(
        perspective_img, H, output_size, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

    return H, birds_eye_view


def calculate_distance(scale_factor, bbox1, bbox2):
    # Extract the coordinates of the bounding boxes
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2

    # Calculate the center points of the bounding boxes
    center1 = np.array([(xmin1 + xmax1) / 2, (ymin1 + ymax1) / 2])
    center2 = np.array([(xmin2 + xmax2) / 2, (ymin2 + ymax2) / 2])

    # Calculate the distance between the center points in pixels
    distance_pixels = np.linalg.norm(center1 - center2)

    # Convert the distance from pixels to meters using the scale factor
    distance_meters = distance_pixels * scale_factor

    return distance_meters


def read_two_meters_distance_xml(xml_file_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    objects = []
    for obj in root.findall("object"):
        obj_dict = {}
        obj_dict["name"] = obj.find("name").text
        obj_dict["id"] = obj.find("id").text
        obj_dict["pose"] = obj.find("pose").text
        obj_dict["truncated"] = obj.find("truncated").text
        obj_dict["difficult"] = obj.find("difficult").text
        bndbox = obj.find("bndbox")
        obj_dict["xmin"] = bndbox.find("xmin").text
        obj_dict["ymin"] = bndbox.find("ymin").text
        obj_dict["xmax"] = bndbox.find("xmax").text
        obj_dict["ymax"] = bndbox.find("ymax").text
        objects.append(obj_dict)

    return objects


def make_bounding_box(obj):
    # Extract the coordinates of the bounding box from the object
    xmin = int(obj["xmin"])
    ymin = int(obj["ymin"])
    xmax = int(obj["xmax"])
    ymax = int(obj["ymax"])

    # Return the bounding box as a tuple
    return (xmin, ymin, xmax, ymax)


def transform_bounding_boxes(bboxes, H, scale_factor):
    # Convert the bounding boxes from (xmin, ymin, xmax, ymax) format to (xmin, ymin, width, height) format
    bboxes = [
        (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]) for bbox in bboxes
    ]

    # Transform the bounding boxes to the bird's-eye view coordinates
    bboxes = cv2.perspectiveTransform(
        np.array([bboxes], dtype=np.float32).transpose(0, 2, 1), H
    )

    # Convert the bounding boxes back to (xmin, ymin, xmax, ymax) format
    bboxes = [
        (int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        for bbox in np.transpose(bboxes[0])
    ]

    # Scale the bounding boxes to the correct size in the bird's-eye view
    bboxes = [
        (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])) for bbox in bboxes
    ]

    return bboxes


def calculate_distance_between_bounding_boxes(bbox1, bbox2, H, scale_factor):
    # Compute the middle bottom points of the bounding boxes
    point1 = ((bbox1[0] + bbox1[2]) / 2, bbox1[3])
    point2 = ((bbox2[0] + bbox2[2]) / 2, bbox2[3])

    # Transform the points to the bird's-eye view coordinates
    point1 = cv2.perspectiveTransform(np.array([[point1]], dtype=np.float32), H)[0][0]
    point2 = cv2.perspectiveTransform(np.array([[point2]], dtype=np.float32), H)[0][0]
    # Convert the points to numpy arrays and take the difference between them
    point1 = np.array(point1, dtype=np.float32)
    point2 = np.array(point2, dtype=np.float32)
    diff = np.subtract(point1, point2)

    # Calculate the distance between the points
    distance = np.linalg.norm(diff)

    # Scale the distance to the correct size in the bird's-eye view
    distance *= scale_factor

    return distance


def plot_point(img, point, color=(0, 255, 0)):
    """Overlays a circle on the image at the specified point with the specified color."""
    cv2.circle(img, (int(point[0]), int(point[1])), 5, color, -1)


def plot_bounding_box(img, bbox, color=(0, 255, 0)):
    xmin, ymin, xmax, ymax = bbox
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness=2)


def calculate_homography(img_path):

    # Load the perspective image
    perspective_img = cv2.imread(img_path)

    # # Generate the bird's-eye view of the perspective image
    H, birds_eye_view = generate_birds_eye_view(perspective_img)

    # # Compute the scale factor between the perspective and bird's-eye views
    src_points = select_points(perspective_img)
    dst_points = select_points(birds_eye_view)
    scale_factor = compute_scale_factor(src_points, dst_points, 2.0)

    return H, scale_factor


def pretty_print_matrix(matrix):
    print(np.array2string(matrix, formatter={"float_kind": lambda x: "%.3f" % x}))
