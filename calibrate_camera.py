from frigate.calibration_utils import (
    generate_birds_eye_view,
    select_points,
    compute_scale_factor,
    pretty_print_matrix,
)
import cv2


img_path = "calibration_test_image.png"


def main():

    perspective_img = cv2.imread(img_path)

    # # Generate the bird's-eye view of the perspective image
    H, birds_eye_view = generate_birds_eye_view(perspective_img)

    # # Compute the scale factor between the perspective and bird's-eye views
    src_points = select_points(perspective_img)
    dst_points = select_points(birds_eye_view)
    scale_factor = compute_scale_factor(src_points, dst_points, 2.0)
    print("Matrix H:")
    pretty_print_matrix(H)
    print("Scale factor:")
    print(scale_factor)


if __name__ == "__main__":
    main()
