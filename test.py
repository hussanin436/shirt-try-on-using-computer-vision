import cv2
import numpy as np


# Function to crop the extra white background by finding the bounding rectangle
# Function to crop the sections with the greatest white pixel density

# Function to resize an image to match the size of another image
def resize_to_match_size(img, reference_img):
    # Get the dimensions of the reference image
    height, width = reference_img.shape[:2]

    # Resize the image to match the reference image's dimensions
    resized_img = cv2.resize(img, (width, height))

    return resized_img


# Function to remove white background by making it transparent
def remove_white_background(img):
    # Check if the image has an alpha channel (transparency)
    if img.shape[2] == 3:  # If there are only 3 channels (BGR), add an alpha channel
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    # Define the color threshold for detecting white pixels
    lower_white = np.array([200, 200, 200, 255])  # Lower threshold for white
    upper_white = np.array([255, 255, 255, 255])  # Upper threshold for white

    # Create a mask that identifies white areas
    white_mask = cv2.inRange(img, lower_white, upper_white)

    # Replace white areas with transparency (set alpha to 0)
    img[white_mask > 0] = (255, 255, 255, 0)

    return img


# Load the images
img1 = cv2.imread('check1.jpg', cv2.IMREAD_UNCHANGED)  # Image to process
img2 = cv2.imread('Shirts/2.png', cv2.IMREAD_UNCHANGED)  # Reference image for resizing


# Step 2: Resize the cropped image to match the size of the reference image
resized_img = resize_to_match_size(img1, img2)

# Step 3: Remove the white background (make transparent)
final_img = remove_white_background(resized_img)

# Save the result
cv2.imwrite('Shirts/0.png', final_img)
print(" Image succesfully resize and remove White background also transfer to his currnt position  ")
# # Display the processed image
# cv2.imshow('Processed Image', final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
