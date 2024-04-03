import cv2
import numpy as np
import os

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply histogram equalization for better contrast
    equalized = cv2.equalizeHist(gray)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    return blurred

def detect_and_compute(image):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(descriptors1, descriptors2):
    if descriptors1 is None or descriptors2 is None:
        return 0
    
    # FLANN parameters
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Lowe's ratio test
    good_matches = []
    if len(matches) > 1:
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

    return len(good_matches)


def search_for_image(target_image, root_folder_path):
    # Convert the target image to grayscale and preprocess it
    target_image_gray = preprocess_image(target_image)
    target_keypoints, target_descriptors = detect_and_compute(target_image_gray)

    for root, dirs, files in os.walk(root_folder_path):
        for dir_name in dirs:
            folder_path = os.path.join(root, dir_name)
            print(f"Searching in {folder_path}...")
            for file_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, file_name)
                # Read the image, convert it to grayscale, and preprocess it
                image = cv2.imread(image_path)
                if image is None:
                    continue
                image_gray = preprocess_image(image)
                keypoints, descriptors = detect_and_compute(image_gray)
                match_count = match_features(target_descriptors, descriptors)
                # Check if enough matches found
                if match_count > 10:  # Adjust this threshold as needed
                    print(f"The card scanned is found at: {image_path}")
                    return

def main():
    root_folder_path = "C:\\Users\\pahar\\OneDrive\\Desktop\\OnePieceOCR\\images"
    cap = cv2.VideoCapture(0)

    card_found = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame.")
            break

        cv2.imshow('Live Scan', frame)

        # Check if a card has already been found
        if not card_found:
            # Search for the image only if a card has not been found yet
            search_for_image(frame, root_folder_path)
            card_found = True  # Set the flag to True after finding the first card

        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

