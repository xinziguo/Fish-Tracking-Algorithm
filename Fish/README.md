# Fish Behavior Tracking & Analysis

This project implements a computer vision-based algorithm to detect, track, and analyze the behavior of Crucian carp in experimental water tanks. It utilizes **HSV color space segmentation** and **geometric analysis** to determine the fish's position and head-tail orientation.

## 🔬 Technical Methodology

The tracking pipeline consists of three main stages: body extraction, feature selection, and orientation correction.

### 1. Fish Body Extraction via HSV

**Objective:** To isolate the fish subject from the background environment using color thresholding.

- **Color Conversion:** Convert frames from **RGB** to **HSV** color space using `cv2.cvtColor` to handle lighting variations better.
  
- **Thresholding:** Define specific color ranges to extract the fish body (e.g., for blue-tagged subjects).
  - **Lower Bound:** `np.array([90, 50, 50])`
  - **Upper Bound:** `np.array([100, 255, 170])`
  > **Note:** These HSV values are based on experimental observations and should be adjusted according to your specific video data source.

- **Masking & ROI:** - Generate a binary mask using `cv2.inRange()`.
  - Apply a **Region of Interest (ROI)** based on the tank's symmetry to filter out external noise using logical "AND" operations.

### 2. Geometric Feature Analysis

**Objective:** To identify the fish's shape and locate key body parts (head/tail).

- **Contour Detection:** Use `cv2.findContours` to detect objects in the mask. The contour with the **largest area** is identified as the target fish.

- **Minimum Enclosing Rectangle:** - Calculate the minimum area rectangle using `cv2.minAreaRect`.
  - Extract geometric parameters: `center`, `width`, `height`, and `angle`.
  - Convert rectangle vertices to coordinates using `cv2.boxPoints`.

- **Feature Point Identification:** - Calculate the **Centroid** of the fish using image moments (`cv2.moments`).
  - Determine the **Head** (assumed furthest point from centroid) and **Tail** (assumed nearest point) on the contour.

### 3. Orientation Correction & Anomaly Detection

**Objective:** To distinguish the head from the tail accurately and correct orientation errors (e.g., 180° flips).

- **Size-End Analysis:** Determine the radius based on the rectangle's dimensions (`np.sqrt(width**2 + height**2) / 2`) to set a reasonable range for feature points.

- **Trajectory Continuity Check:** Calculate the Euclidean distance between the **current Head** and the **previous frame's Tail**:
  $$\text{Distance} = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}$$
  - **Logic:** If the current head is suspiciously close to the previous tail (indicating an impossible instant flip), the algorithm detects an anomaly.

- **Correction Strategy:** - The function `correct_fish_orientation()` is triggered to swap the head and tail coordinates if an anomaly is detected.
  - Corrected coordinates are logged and used to plot the final **movement trajectory** and **heatmap**.
