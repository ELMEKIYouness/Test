import streamlit as st
import numpy as np
import cv2

# Define your segmentation functions here
def apply_thresholding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return thresh

def apply_edge_detection(image):
    edges = cv2.Canny(image, 100, 200)
    return edges

def apply_watershed(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 0] = 0
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]
    return image

def apply_kmeans(image):
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)
    # Define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 8
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((image.shape))
    return res2

def apply_active_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    return image

st.title('Face Segmenter')
st.write("Upload an image and see the results of different segmentation techniques.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Read the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        st.image(img, caption='Uploaded Image.', use_column_width=True)

        # Apply and display various segmentation techniques
        st.write("Thresholding")
        thresh = apply_thresholding(img)
        st.image(thresh, caption='Thresholding', use_column_width=True)

        st.write("Edge Detection (Canny)")
        edges = apply_edge_detection(img)
        st.image(edges, caption='Edge Detection (Canny)', use_column_width=True)

        st.write("Region-based Segmentation (Watershed)")
        watershed_img = apply_watershed(img.copy())
        st.image(watershed_img, caption='Watershed', use_column_width=True)

        st.write("Clustering (K-means)")
        kmeans_img = apply_kmeans(img)
        st.image(kmeans_img, caption='K-means', use_column_width=True)

        st.write("Active Contours")
        active_contours_img = apply_active_contours(img)
        st.image(active_contours_img, caption='Active Contours', use_column_width=True)

    except Exception as e:
        st.write("Error occurred while processing the image:", e)
