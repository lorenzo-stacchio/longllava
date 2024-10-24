import cv2
import numpy as np
import os
# Load the image

def plot_caption(image_path,caption):
    basename = os.path.basename(image_path).split(".")[0]
    image = cv2.imread(image_path)

    # Set the canvas dimensions
    canvas_height = image.shape[0]
    canvas_width = image.shape[1] + 400  # Adding space for text

    # Create a white canvas
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

    # Place the image on the left side of the canvas
    canvas[:, :image.shape[1]] = image

    # Set font, scale, and thickness for the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2
    text_color = (0, 0, 0)  # Black text

    # Calculate position for the caption (center it vertically)
    text_x = image.shape[1] + 30
    text_y = canvas_height // 2

    # Add caption text to the right side of the canvas
    cv2.putText(canvas, caption, (text_x, text_y), font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)

    # Optionally, add a cool effect like a border or shadow
    border_color = (0, 0, 255)  # Red border
    cv2.rectangle(canvas, (0, 0), (image.shape[1], canvas_height), border_color, 10)

    # Show the resulting image
    # cv2.imshow('Image with Caption', canvas)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Save the output
    cv2.imwrite(f'out_images/{basename}_caption.jpg', canvas)
