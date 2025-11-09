import cv2
import numpy as np
import os

def get_scaled_display_info(image, max_height=1200):
    (h_orig, w_orig) = image.shape[:2]
    r = max_height / h_orig
    dim = (int(w_orig * r), max_height) 
    display_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return display_image, r, dim

def click_event(event, x, y, flag, param_state):
    points = param_state['points']
    image_display = param_state['image_display']
    window_name = param_state['window_name']

    def redraw_points():
        image_copy = image_display.copy()
        for i, p in enumerate(points):
            cv2.circle(image_copy, p, 5, (0, 255, 0), -1)
            if i > 0:
                cv2.line(image_copy, points[i-1], p, (255, 0, 0), 2)
        if len(points) == 4:
            cv2.line(image_copy, points[3], points[0], (255, 0, 0), 2)
        cv2.imshow(window_name, image_copy)

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            redraw_points()

def get_rotated_points_from_user(image_original, max_display_height=1200):
    image_display, r, dim = get_scaled_display_info(image_original, max_display_height)
    
    window_name = 'ooooooooooo'
    
    state = {
        'points': [],
        'image_display': image_display,
        'window_name': window_name
    }
    
    cv2.namedWindow(window_name)
    cv2.resizeWindow(window_name, dim[0], dim[1])
    cv2.setMouseCallback(window_name, click_event, param=state) 
    cv2.imshow(window_name, image_display)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)

    points_display_np = np.array(state['points'], dtype=np.float32)
    points_orig = (points_display_np / r).astype(np.int32)
    
    return points_orig, r

def apply_masked_blur(image, points_orig, blur_kernel=(130, 130)):
    (h_orig, w_orig) = image.shape[:2]
    
    mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
    pts_for_fill = points_orig.reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts_for_fill], 255)
    
    image_blurred = cv2.blur(image, blur_kernel) 
    
    image_result = image.copy()
    image_result[mask == 255] = image_blurred[mask == 255] 
    
    return image_result

def main():
    path = './img/motor.tif'
    max_height = 1200
    blur_kernel_size = (130, 130)

    image = cv2.imread(path)

    points_orig, scale_ratio = get_rotated_points_from_user(image, max_height)
    
    if points_orig is None: 
        return

    result_image = apply_masked_blur(image, points_orig, blur_kernel_size)
    
    window_name_result = 'ckck is fxxk rich man'
    display_result_image, _, dim_result = get_scaled_display_info(result_image, max_height)
    
    cv2.namedWindow(window_name_result) 
    cv2.resizeWindow(window_name_result, dim_result[0], dim_result[1]) 
    cv2.imshow(window_name_result, display_result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    output_filename = 'blurred_motor.tif'
    output_path = os.path.join(os.path.dirname(path), output_filename)
    
    cv2.imwrite(output_path, result_image) 

if __name__ == '__main__':
    main()