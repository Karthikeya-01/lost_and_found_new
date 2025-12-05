"""
Image Feature Extraction Module
File: src/feature_extraction/image_features.py

This module extracts various features from images:
- Color features (histograms, dominant colors, color moments)
- Texture features (LBP, GLCM)
- Shape features (edges, contours)
- SIFT keypoints (optional)
"""

# import cv2
# import numpy as np
# from PIL import Image
# from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
# from sklearn.cluster import KMeans
# import warnings
# warnings.filterwarnings('ignore')


# class ImageFeatureExtractor:
#     """Extract comprehensive features from images for matching"""

#     def __init__(self, target_size=(224, 224)):
#         """
#         Initialize feature extractor
        
#         Args:
#             target_size: Resize images to this size for consistent features
#         """
#         self.target_size = target_size

#     def load_and_preprocess(self, image_path):
#         """
#         Load and preprocess image
        
#         Args:
#             image_path: Path to image file
            
#         Returns:
#             Preprocessed image in RGB and grayscale
#         """
#         # Load image
#         img = cv2.imread(str(image_path))
#         if img is None:
#             raise ValueError(f"Could not load image: {image_path}")

#         # Convert BGR to RGB
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         # Resize
#         img_rgb = cv2.resize(img_rgb, self.target_size)

#         # Create grayscale version
#         img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

#         return img_rgb, img_gray

#     def extract_color_histogram(self, img_rgb, bins=32):
#         """
#         Extract color histogram features in HSV space
        
#         Args:
#             img_rgb: RGB image
#             bins: Number of bins per channel
            
#         Returns:
#             Flattened histogram features
#         """
#         # Convert to HSV (better for color comparison)
#         img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

#         # Calculate histogram for each channel
#         hist_h = cv2.calcHist([img_hsv], [0], None, [bins], [0, 180])
#         hist_s = cv2.calcHist([img_hsv], [1], None, [bins], [0, 256])
#         hist_v = cv2.calcHist([img_hsv], [2], None, [bins], [0, 256])

#         # Normalize
#         hist_h = cv2.normalize(hist_h, hist_h).flatten()
#         hist_s = cv2.normalize(hist_s, hist_s).flatten()
#         hist_v = cv2.normalize(hist_v, hist_v).flatten()

#         # Concatenate
#         color_hist = np.concatenate([hist_h, hist_s, hist_v])

#         return color_hist

#     def extract_dominant_colors(self, img_rgb, n_colors=5):
#         """
#         Extract dominant colors using K-means clustering
        
#         Args:
#             img_rgb: RGB image
#             n_colors: Number of dominant colors to extract
            
#         Returns:
#             Dominant color features (flattened RGB values + percentages)
#         """
#         # Reshape image to be a list of pixels
#         pixels = img_rgb.reshape(-1, 3)

#         # Perform k-means clustering
#         kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
#         kmeans.fit(pixels)

#         # Get colors and their frequencies
#         colors = kmeans.cluster_centers_
#         labels = kmeans.labels_

#         # Calculate percentage of each color
#         unique, counts = np.unique(labels, return_counts=True)
#         percentages = counts / len(labels)

#         # Sort by percentage
#         sorted_idx = np.argsort(percentages)[::-1]
#         colors = colors[sorted_idx]
#         percentages = percentages[sorted_idx]

#         # Flatten: [R1, G1, B1, P1, R2, G2, B2, P2, ...]
#         features = []
#         for color, percentage in zip(colors, percentages):
#             features.extend(color)
#             features.append(percentage)

#         return np.array(features)

#     def extract_color_moments(self, img_rgb):
#         """
#         Extract color moments (mean, std, skewness) for each channel
        
#         Args:
#             img_rgb: RGB image
            
#         Returns:
#             Color moment features (9 values: 3 moments x 3 channels)
#         """
#         features = []

#         for channel in range(3):
#             channel_data = img_rgb[:, :, channel].flatten()

#             # Mean
#             mean = np.mean(channel_data)
#             # Standard deviation
#             std = np.std(channel_data)
#             # Skewness
#             skewness = np.mean(((channel_data - mean) / std)
#                                ** 3) if std > 0 else 0

#             features.extend([mean, std, skewness])

#         return np.array(features)

#     def extract_lbp_features(self, img_gray, n_points=24, radius=3):
#         """
#         Extract Local Binary Pattern (LBP) texture features
        
#         Args:
#             img_gray: Grayscale image
#             n_points: Number of circularly symmetric neighbor points
#             radius: Radius of circle
            
#         Returns:
#             LBP histogram features
#         """
#         # Compute LBP
#         lbp = local_binary_pattern(
#             img_gray, n_points, radius, method='uniform')

#         # Calculate histogram
#         n_bins = n_points + 2  # For uniform LBP
#         hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))

#         # Normalize
#         hist = hist.astype(float)
#         hist /= (hist.sum() + 1e-7)

#         return hist

#     def extract_glcm_features(self, img_gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
#         """
#         Extract GLCM (Gray-Level Co-occurrence Matrix) texture features
        
#         Args:
#             img_gray: Grayscale image
#             distances: Pixel pair distance offsets
#             angles: Pixel pair angles
            
#         Returns:
#             GLCM texture properties
#         """
#         # Compute GLCM
#         glcm = graycomatrix(img_gray, distances=distances, angles=angles,
#                             levels=256, symmetric=True, normed=True)

#         # Extract properties
#         features = []
#         properties = ['contrast', 'dissimilarity',
#                       'homogeneity', 'energy', 'correlation']

#         for prop in properties:
#             values = graycoprops(glcm, prop)
#             # Average across distances and angles
#             features.append(np.mean(values))
#             features.append(np.std(values))   # Std across distances and angles

#         return np.array(features)

#     def extract_edge_features(self, img_gray):
#         """
#         Extract edge-based features using Canny edge detection
        
#         Args:
#             img_gray: Grayscale image
            
#         Returns:
#             Edge density and distribution features
#         """
#         # Apply Canny edge detection
#         edges = cv2.Canny(img_gray, 100, 200)

#         # Calculate features
#         edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

#         # Edge distribution (divide image into grid)
#         grid_size = 4
#         h, w = edges.shape
#         grid_h, grid_w = h // grid_size, w // grid_size

#         grid_features = []
#         for i in range(grid_size):
#             for j in range(grid_size):
#                 grid = edges[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
#                 grid_density = np.sum(grid > 0) / (grid_h * grid_w)
#                 grid_features.append(grid_density)

#         features = [edge_density] + grid_features
#         return np.array(features)

#     def extract_shape_features(self, img_gray):
#         """
#         Extract shape-based features using contours
        
#         Args:
#             img_gray: Grayscale image
            
#         Returns:
#             Shape features (contour count, average area, etc.)
#         """
#         # Threshold image
#         _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

#         # Find contours
#         contours, _ = cv2.findContours(
#             thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         if len(contours) == 0:
#             return np.zeros(6)

#         # Extract features
#         n_contours = len(contours)
#         areas = [cv2.contourArea(c) for c in contours]
#         perimeters = [cv2.arcLength(c, True) for c in contours]

#         # Compute statistics
#         avg_area = np.mean(areas) if areas else 0
#         avg_perimeter = np.mean(perimeters) if perimeters else 0

#         # Largest contour features
#         largest_contour = max(contours, key=cv2.contourArea)
#         largest_area = cv2.contourArea(largest_contour)
#         largest_perimeter = cv2.arcLength(largest_contour, True)

#         # Compactness (circularity)
#         compactness = (4 * np.pi * largest_area) / \
#             (largest_perimeter ** 2) if largest_perimeter > 0 else 0

#         features = [
#             n_contours,
#             avg_area,
#             avg_perimeter,
#             largest_area,
#             largest_perimeter,
#             compactness
#         ]

#         return np.array(features)

#     def extract_sift_features(self, img_gray, n_features=100):
#         """
#         Extract SIFT keypoint features (optional - can be computationally expensive)
        
#         Args:
#             img_gray: Grayscale image
#             n_features: Number of keypoints to detect
            
#         Returns:
#             SIFT descriptor statistics
#         """
#         try:
#             # Create SIFT detector
#             sift = cv2.SIFT_create(nfeatures=n_features)

#             # Detect and compute
#             keypoints, descriptors = sift.detectAndCompute(img_gray, None)

#             if descriptors is None or len(descriptors) == 0:
#                 return np.zeros(128)  # SIFT descriptor size

#             # Aggregate descriptors (mean pooling)
#             features = np.mean(descriptors, axis=0)

#             return features
#         except:
#             # SIFT not available in some OpenCV builds
#             return np.zeros(128)

#     def extract_all_features(self, image_path, include_sift=False):
#         """
#         Extract all features from an image
        
#         Args:
#             image_path: Path to image file
#             include_sift: Whether to include SIFT features (slower)
            
#         Returns:
#             Dictionary with all features
#         """
#         # Load image
#         img_rgb, img_gray = self.load_and_preprocess(image_path)

#         # Extract all features
#         features = {
#             'color_histogram': self.extract_color_histogram(img_rgb),
#             'dominant_colors': self.extract_dominant_colors(img_rgb),
#             'color_moments': self.extract_color_moments(img_rgb),
#             'lbp': self.extract_lbp_features(img_gray),
#             'glcm': self.extract_glcm_features(img_gray),
#             'edges': self.extract_edge_features(img_gray),
#             'shapes': self.extract_shape_features(img_gray),
#         }

#         if include_sift:
#             features['sift'] = self.extract_sift_features(img_gray)

#         # Flatten all features into a single vector
#         feature_vector = np.concatenate(
#             [v.flatten() for v in features.values()])

#         return {
#             'features': features,
#             'feature_vector': feature_vector,
#             'feature_names': list(features.keys()),
#             'feature_dims': {k: len(v.flatten()) for k, v in features.items()}
#         }

#     def get_feature_vector_size(self, include_sift=False):
#         """Get the total size of the feature vector"""
#         sizes = {
#             'color_histogram': 96,  # 32 bins * 3 channels
#             'dominant_colors': 20,  # 5 colors * (3 RGB + 1 percentage)
#             'color_moments': 9,     # 3 moments * 3 channels
#             'lbp': 26,              # For n_points=24
#             'glcm': 10,             # 5 properties * 2 (mean, std)
#             'edges': 17,            # 1 overall + 16 grid
#             'shapes': 6,            # 6 shape features
#         }

#         if include_sift:
#             sizes['sift'] = 128

#         return sum(sizes.values())


# # Example usage and testing
# if __name__ == "__main__":
#     import sys
#     from pathlib import Path

#     # Test the feature extractor
#     extractor = ImageFeatureExtractor()

#     print("="*80)
#     print("IMAGE FEATURE EXTRACTOR TEST")
#     print("="*80)

#     # Expected feature vector size
#     print(
#         f"\nFeature Vector Size (without SIFT): {extractor.get_feature_vector_size(False)}")
#     print(
#         f"Feature Vector Size (with SIFT): {extractor.get_feature_vector_size(True)}")

#     # Test on a sample image if path provided
#     if len(sys.argv) > 1:
#         image_path = sys.argv[1]
#         print(f"\nTesting on: {image_path}")

#         try:
#             result = extractor.extract_all_features(
#                 image_path, include_sift=False)

#             print("\n" + "-"*80)
#             print("EXTRACTED FEATURES:")
#             print("-"*80)

#             for name, dim in result['feature_dims'].items():
#                 print(f"  {name:.<25} {dim} dimensions")

#             print(
#                 f"\n  Total feature vector: {len(result['feature_vector'])} dimensions")

#             print("\n" + "-"*80)
#             print("SAMPLE FEATURE VALUES:")
#             print("-"*80)

#             # Show first few values of each feature type
#             for name, features in result['features'].items():
#                 flat = features.flatten()
#                 print(f"\n  {name}:")
#                 print(f"    Shape: {features.shape}")
#                 print(f"    First 5 values: {flat[:5]}")
#                 print(
#                     f"    Mean: {np.mean(flat):.4f}, Std: {np.std(flat):.4f}")

#             print("\n✓ Feature extraction successful!")

#         except Exception as e:
#             print(f"\n✗ Error: {e}")
#     else:
#         print("\nTo test on an image, run:")
#         print("  python image_features.py path/to/image.jpg")

import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


def get_cv2():
    import cv2
    return cv2


class ImageFeatureExtractor:
    """Extract comprehensive features from images for matching"""

    def __init__(self, target_size=(224, 224)):
        self.cv2 = get_cv2()      # <-- store cv2 instance
        self.target_size = target_size

    def load_and_preprocess(self, image_path):
        img = self.cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        img_rgb = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2RGB)
        img_rgb = self.cv2.resize(img_rgb, self.target_size)
        img_gray = self.cv2.cvtColor(img_rgb, self.cv2.COLOR_RGB2GRAY)

        return img_rgb, img_gray

    def extract_color_histogram(self, img_rgb, bins=32):
        img_hsv = self.cv2.cvtColor(img_rgb, self.cv2.COLOR_RGB2HSV)

        hist_h = self.cv2.calcHist([img_hsv], [0], None, [bins], [0, 180])
        hist_s = self.cv2.calcHist([img_hsv], [1], None, [bins], [0, 256])
        hist_v = self.cv2.calcHist([img_hsv], [2], None, [bins], [0, 256])

        hist_h = self.cv2.normalize(hist_h, hist_h).flatten()
        hist_s = self.cv2.normalize(hist_s, hist_s).flatten()
        hist_v = self.cv2.normalize(hist_v, hist_v).flatten()

        return np.concatenate([hist_h, hist_s, hist_v])

    def extract_dominant_colors(self, img_rgb, n_colors=5):
        pixels = img_rgb.reshape(-1, 3)
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)

        colors = kmeans.cluster_centers_
        labels = kmeans.labels_

        unique, counts = np.unique(labels, return_counts=True)
        percentages = counts / len(labels)

        sorted_idx = np.argsort(percentages)[::-1]
        colors = colors[sorted_idx]
        percentages = percentages[sorted_idx]

        features = []
        for color, pct in zip(colors, percentages):
            features.extend(color)
            features.append(pct)

        return np.array(features)

    def extract_color_moments(self, img_rgb):
        features = []
        for c in range(3):
            data = img_rgb[:, :, c].flatten()
            mean = np.mean(data)
            std = np.std(data)
            skew = np.mean(((data - mean) / std)**3) if std > 0 else 0
            features.extend([mean, std, skew])
        return np.array(features)

    def extract_lbp_features(self, img_gray, n_points=24, radius=3):
        lbp = local_binary_pattern(
            img_gray, n_points, radius, method='uniform')
        hist, _ = np.histogram(
            lbp.ravel(), bins=n_points+2, range=(0, n_points+2))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-7)
        return hist

    def extract_glcm_features(self, img_gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
        glcm = graycomatrix(img_gray, distances=distances, angles=angles,
                            levels=256, symmetric=True, normed=True)

        props = ['contrast', 'dissimilarity',
                 'homogeneity', 'energy', 'correlation']
        features = []
        for p in props:
            vals = graycoprops(glcm, p)
            features.append(np.mean(vals))
            features.append(np.std(vals))
        return np.array(features)

    def extract_edge_features(self, img_gray):
        edges = self.cv2.Canny(img_gray, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size

        grid_size = 4
        h, w = edges.shape
        gh, gw = h // grid_size, w // grid_size

        grid_features = []
        for i in range(grid_size):
            for j in range(grid_size):
                grid = edges[i*gh:(i+1)*gh, j*gw:(j+1)*gw]
                grid_features.append(np.sum(grid > 0) / (gh * gw))

        return np.array([edge_density] + grid_features)

    def extract_shape_features(self, img_gray):
        _, thresh = self.cv2.threshold(
            img_gray, 127, 255, self.cv2.THRESH_BINARY)
        contours, _ = self.cv2.findContours(
            thresh, self.cv2.RETR_EXTERNAL, self.cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return np.zeros(6)

        areas = [self.cv2.contourArea(c) for c in contours]
        perimeters = [self.cv2.arcLength(c, True) for c in contours]

        avg_area = np.mean(areas)
        avg_perim = np.mean(perimeters)

        largest = max(contours, key=self.cv2.contourArea)
        largest_area = self.cv2.contourArea(largest)
        largest_perim = self.cv2.arcLength(largest, True)

        compactness = (4 * np.pi * largest_area) / \
            (largest_perim ** 2) if largest_perim > 0 else 0

        return np.array([
            len(contours),
            avg_area,
            avg_perim,
            largest_area,
            largest_perim,
            compactness
        ])

    def extract_sift_features(self, img_gray, n_features=100):
        try:
            sift = self.cv2.SIFT_create(nfeatures=n_features)
            kpts, desc = sift.detectAndCompute(img_gray, None)
            if desc is None or len(desc) == 0:
                return np.zeros(128)
            return np.mean(desc, axis=0)
        except:
            return np.zeros(128)

    def extract_all_features(self, image_path, include_sift=False):
        img_rgb, img_gray = self.load_and_preprocess(image_path)

        features = {
            'color_histogram': self.extract_color_histogram(img_rgb),
            'dominant_colors': self.extract_dominant_colors(img_rgb),
            'color_moments': self.extract_color_moments(img_rgb),
            'lbp': self.extract_lbp_features(img_gray),
            'glcm': self.extract_glcm_features(img_gray),
            'edges': self.extract_edge_features(img_gray),
            'shapes': self.extract_shape_features(img_gray),
        }

        if include_sift:
            features['sift'] = self.extract_sift_features(img_gray)

        vec = np.concatenate([v.flatten() for v in features.values()])

        return {
            'features': features,
            'feature_vector': vec,
            'feature_names': list(features.keys()),
            'feature_dims': {k: len(v.flatten()) for k, v in features.items()},
        }
    def get_feature_vector_size(self, include_sift=False):
        """Get the total size of the feature vector"""
        sizes = {
            'color_histogram': 96,  # 32 bins * 3 channels
            'dominant_colors': 20,  # 5 colors * (3 RGB + 1 percentage)
            'color_moments': 9,     # 3 moments * 3 channels
            'lbp': 26,              # For n_points=24
            'glcm': 10,             # 5 properties * 2 (mean, std)
            'edges': 17,            # 1 overall + 16 grid
            'shapes': 6,            # 6 shape features
        }

        if include_sift:
            sizes['sift'] = 128

        return sum(sizes.values())


# Example usage and testing
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Test the feature extractor
    extractor = ImageFeatureExtractor()

    print("="*80)
    print("IMAGE FEATURE EXTRACTOR TEST")
    print("="*80)

    # Expected feature vector size
    print(
        f"\nFeature Vector Size (without SIFT): {extractor.get_feature_vector_size(False)}")
    print(
        f"Feature Vector Size (with SIFT): {extractor.get_feature_vector_size(True)}")

    # Test on a sample image if path provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"\nTesting on: {image_path}")

        try:
            result = extractor.extract_all_features(
                image_path, include_sift=False)

            print("\n" + "-"*80)
            print("EXTRACTED FEATURES:")
            print("-"*80)

            for name, dim in result['feature_dims'].items():
                print(f"  {name:.<25} {dim} dimensions")

            print(
                f"\n  Total feature vector: {len(result['feature_vector'])} dimensions")

            print("\n" + "-"*80)
            print("SAMPLE FEATURE VALUES:")
            print("-"*80)

            # Show first few values of each feature type
            for name, features in result['features'].items():
                flat = features.flatten()
                print(f"\n  {name}:")
                print(f"    Shape: {features.shape}")
                print(f"    First 5 values: {flat[:5]}")
                print(
                    f"    Mean: {np.mean(flat):.4f}, Std: {np.std(flat):.4f}")

            print("\n✓ Feature extraction successful!")

        except Exception as e:
            print(f"\n✗ Error: {e}")
    else:
        print("\nTo test on an image, run:")
        print("  python image_features.py path/to/image.jpg")
