import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import PIL
import cv2 
import numpy as np 
import traceback 
import argparse 
import csv

# --- Added Image Processing Functions ---

def gamma_correct_target_d(image, target):
    """
    Perform gamma correction on the image to match the target mean intensity
    Uses dichotomy to find the gamma value
    """
    image_float = image.astype(np.float32)
    init_mean = np.mean(image_float)

    # Determine search direction based on target vs. initial mean
    if init_mean == target:
        return image, 1.0

    # Define lower and upper bounds for gamma
    if init_mean < target:
        # Need to decrease gamma (brighten image)
        gamma_min = 0.1
        gamma_max = 1.0
    else:
        # Need to increase gamma (darken image)
        gamma_min = 1.0
        gamma_max = 5.0 # Increased upper bound for potentially darker images

    # Dichotomy search
    max_iterations = 20
    tolerance = 0.5 # Tolerance for mean comparison
    gamma = 1.0 # Default gamma
    corr_image = image # Default corrected image

    for _ in range(max_iterations):
        gamma = (gamma_min + gamma_max) / 2

        # Apply gamma correction
        # Avoid division by zero if image_float contains zeros
        # Add epsilon to prevent log(0) issues if image contains black pixels
        epsilon = 1e-6
        corr_image_norm = np.power((image_float + epsilon) / 255.0, gamma)
        corr_image = np.clip(corr_image_norm * 255, 0, 255).astype(np.uint8)

        current_mean = np.mean(corr_image)

        # Check if we're close enough to target
        if abs(current_mean - target) < tolerance:
            break

        # Adjust search range based on comparison with target
        if init_mean < target: # Brightening
            if current_mean < target:
                gamma_max = gamma # Need less brightening (higher gamma)
            else:
                gamma_min = gamma # Need more brightening (lower gamma)
        else: # Darkening
            if current_mean < target:
                gamma_max = gamma # Need less darkening (lower gamma)
            else:
                gamma_min = gamma # Need more darkening (higher gamma)


    # Ensure the final image is returned even if loop finishes early
    # Re-apply the best gamma found if needed (or just use the last corr_image)
    # corr_image_norm = np.power(image_float / 255.0, gamma)
    # corr_image = np.clip(corr_image_norm * 255, 0, 255).astype(np.uint8)

    return corr_image, gamma


def extract_cells_infiltrations_masks(image, threshold=100, gray_morph_iterations=5, white_threshold=175): # Added white_threshold parameter
    # Ensure image is 3-channel BGR for consistency
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4: # Handle RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    light_pixels = (image_grayscale >= threshold)
    mask = np.zeros_like(image_grayscale) # Start with black mask
    mask[light_pixels] = 255 # Set light pixels to white

    # apply morphological operations to remove noise
    small_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    medium_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    large_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_filtered = mask.copy()
    # Use iterations consistent with notebook
    mask_filtered = cv2.morphologyEx(mask_filtered, cv2.MORPH_OPEN, kernel=medium_ellipse, iterations=2)
    mask_filtered = cv2.morphologyEx(mask_filtered, cv2.MORPH_OPEN, kernel=small_ellipse, iterations=4)
    mask_filtered = cv2.morphologyEx(mask_filtered, cv2.MORPH_CLOSE, kernel=large_ellipse, iterations=1)


    # find contours of the mask, filter out contours that are too small
    contours, _ = cv2.findContours(mask_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    new_mask_filtered = np.zeros_like(mask_filtered) # Start with black mask
    for c in contours:
        area = cv2.contourArea(c)
        if area > 800: # Area threshold from notebook
            cv2.drawContours(new_mask_filtered, [c], -1, 255, -1) # Fill the contour

    # Dilate slightly as in notebook
    new_mask_filtered = cv2.dilate(new_mask_filtered, kernel=small_ellipse, iterations=3)


    # find pixels that are not white but are gray (adjust range slightly if needed)
    gray_lower_bound = threshold
    gray_upper_bound = threshold + 50
    only_gray_pixels = (image_grayscale > gray_lower_bound) & (image_grayscale < gray_upper_bound)

    mask_gray = np.zeros_like(image_grayscale)
    mask_gray[only_gray_pixels] = 255

    # apply morphological operations to remove noise from gray mask
    mask_morph_gray = mask_gray.copy()
    small_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) # Define kernels here
    large_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    # Use the parameter for MORPH_OPEN iterations
    mask_morph_gray = cv2.morphologyEx(mask_morph_gray, cv2.MORPH_OPEN, kernel=small_ellipse, iterations=gray_morph_iterations)
    # Keep MORPH_CLOSE iterations fixed for now, or add another parameter
    mask_morph_gray = cv2.morphologyEx(mask_morph_gray, cv2.MORPH_CLOSE, kernel=large_ellipse, iterations=5)


    # Find contours in the area-filtered mask (new_mask_filtered)
    contours, _ = cv2.findContours(new_mask_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create the final validated mask
    validated_mask = np.zeros_like(new_mask_filtered)

    # Check each contour for overlap with the morphological gray mask
    for contour in contours:
        # Create a temporary mask for this contour
        temp_mask = np.zeros_like(new_mask_filtered)
        cv2.drawContours(temp_mask, [contour], 0, 255, -1) # Fill contour

        # Check if there's any overlap with mask_morph_gray
        overlap = cv2.bitwise_and(temp_mask, mask_morph_gray)
        if np.any(overlap > 0): # Check if any pixel in overlap is non-zero
            # Valid contour (has overlap with gray regions)
            cv2.drawContours(validated_mask, [contour], 0, 255, -1) # Add to validated mask

    # --- Create white mask (brightest parts within validated areas) ---
    # Use the white_threshold parameter here
    white_pixels = (image_grayscale >= white_threshold)
    white_mask_raw = np.zeros_like(image_grayscale)
    white_mask_raw[white_pixels] = 255

    # Keep only white pixels that are within the validated mask
    white_mask_final = cv2.bitwise_and(white_mask_raw, validated_mask)

    # validated_mask contains all valid regions (gray + white)
    # white_mask_final contains only the bright white parts within those regions
    return validated_mask, white_mask_final


def visualize_result(image, base_gray_mask, base_white_mask, deleted_contours):
    """ Visualize result with active areas (red/green) and deleted outlines (gray). """
    # Ensure image is BGR
    if len(image.shape) == 2:
        masked_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        masked_image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:
        masked_image = image.copy()

    # Ensure base masks are single channel 8-bit
    if len(base_gray_mask.shape) == 3: base_gray_mask = cv2.cvtColor(base_gray_mask, cv2.COLOR_BGR2GRAY)
    if len(base_white_mask.shape) == 3: base_white_mask = cv2.cvtColor(base_white_mask, cv2.COLOR_BGR2GRAY)
    base_gray_mask = base_gray_mask.astype(np.uint8)
    base_white_mask = base_white_mask.astype(np.uint8)

    # Create active masks by removing deleted contours
    active_gray_mask = base_gray_mask.copy()
    active_white_mask = base_white_mask.copy()
    if deleted_contours: # Avoid error if list is empty
        cv2.drawContours(active_gray_mask, deleted_contours, -1, 0, -1) # Fill deleted areas with black
        cv2.drawContours(active_white_mask, deleted_contours, -1, 0, -1)

    # Create colored overlays for ACTIVE areas
    # Red for active gray areas (active total mask MINUS active white mask)
    active_gray_only_mask = cv2.subtract(active_gray_mask, active_white_mask)
    red_overlay = np.zeros_like(masked_image)
    red_overlay[active_gray_only_mask == 255] = [0, 0, 255] # Red in BGR

    # Green for active white areas
    green_overlay = np.zeros_like(masked_image)
    green_overlay[active_white_mask == 255] = [0, 255, 0] # Green in BGR

    # Combine overlays
    combined_overlay = cv2.add(red_overlay, green_overlay)

    # Add weighted overlay to the original image
    alpha = 0.5 # Transparency factor for overlay
    final_visualization = cv2.addWeighted(masked_image, 1, combined_overlay, alpha, 0)

    # Draw outlines for DELETED contours in gray on the final image
    if deleted_contours:
        # cv2.drawContours(final_visualization, deleted_contours, -1, (255, 128, 128), 2) # Gray outline, thickness 2
        cv2.drawContours(final_visualization, deleted_contours, -1, (0, 0, 255), 2) # Red outline

    return final_visualization


def generate_debug_visualization(image, threshold=100, gray_morph_iterations=5): # Added parameter
    """
    Generates a grid of intermediate processing steps for debugging.
    """
    vis_list = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_color = (255, 255, 255) # White
    line_type = 2

    def add_text(img, text):
        # Ensure image is BGR
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # Add text background rectangle
        text_size, _ = cv2.getTextSize(text, font, font_scale, line_type)
        text_w, text_h = text_size
        cv2.rectangle(img, (0, 0), (text_w + 5, text_h + 10), (0,0,0), -1) # Black background
        # Add text
        cv2.putText(img, text, (5, text_h + 5), font, font_scale, font_color, line_type)
        return img

    # 0. Original (Corrected) Image
    vis_list.append(add_text(image.copy(), "0. Corrected"))

    # --- Steps from extract_cells_infiltrations_masks ---
    if len(image.shape) == 2:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:
        image_bgr = image.copy()

    image_grayscale = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Grayscale
    vis_list.append(add_text(image_grayscale.copy(), "1. Grayscale"))

    # 2. Initial Threshold Mask
    light_pixels = (image_grayscale >= threshold)
    mask = np.zeros_like(image_grayscale)
    mask[light_pixels] = 255
    vis_list.append(add_text(mask.copy(), "2. Initial Mask"))

    # 3. Morphological Filtering (Noise Removal)
    small_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    medium_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    large_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_filtered = mask.copy()
    mask_filtered = cv2.morphologyEx(mask_filtered, cv2.MORPH_OPEN, kernel=medium_ellipse, iterations=2)
    mask_filtered = cv2.morphologyEx(mask_filtered, cv2.MORPH_OPEN, kernel=small_ellipse, iterations=4)
    mask_filtered = cv2.morphologyEx(mask_filtered, cv2.MORPH_CLOSE, kernel=large_ellipse, iterations=1)
    vis_list.append(add_text(mask_filtered.copy(), "3. Filtered Mask"))

    # 4. Area Filtering + Dilation
    contours, _ = cv2.findContours(mask_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    new_mask_filtered = np.zeros_like(mask_filtered)
    contours_vis = cv2.cvtColor(mask_filtered.copy(), cv2.COLOR_GRAY2BGR) # For drawing contours
    for c in contours:
        area = cv2.contourArea(c)
        if area > 800:
            cv2.drawContours(new_mask_filtered, [c], -1, 255, -1)
            cv2.drawContours(contours_vis, [c], -1, (0, 255, 0), 1) # Green for kept
        else:
            cv2.drawContours(contours_vis, [c], -1, (0, 0, 255), 1) # Red for removed
    new_mask_filtered = cv2.dilate(new_mask_filtered, kernel=small_ellipse, iterations=3)
    # vis_list.append(add_text(contours_vis, "4a. Contours (Area)")) # Optional contour vis
    vis_list.append(add_text(new_mask_filtered.copy(), "4. Area Filtered"))

    # 5. Gray Pixels Mask + Morphological Filtering - Convert to BGR
    gray_lower_bound = threshold
    gray_upper_bound = threshold + 50
    only_gray_pixels = (image_grayscale > gray_lower_bound) & (image_grayscale < gray_upper_bound)
    mask_gray = np.zeros_like(image_grayscale)
    mask_gray[only_gray_pixels] = 255
    mask_morph_gray = mask_gray.copy()
    small_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) # Define kernels
    large_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    # Use the parameter for MORPH_OPEN iterations
    mask_morph_gray = cv2.morphologyEx(mask_morph_gray, cv2.MORPH_OPEN, kernel=small_ellipse, iterations=gray_morph_iterations)
    # Keep MORPH_CLOSE iterations fixed
    mask_morph_gray = cv2.morphologyEx(mask_morph_gray, cv2.MORPH_CLOSE, kernel=large_ellipse, iterations=5)
    # Update title text
    vis_list.append(add_text(cv2.cvtColor(mask_morph_gray.copy(), cv2.COLOR_GRAY2BGR), f"5. Filtered Gray ({gray_morph_iterations} iter)"))

    # 6. Validated Mask (Overlap Check) - Convert to BGR
    contours_final, _ = cv2.findContours(new_mask_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    validated_mask = np.zeros_like(new_mask_filtered)
    validation_vis = cv2.cvtColor(image_grayscale.copy(), cv2.COLOR_GRAY2BGR) # Base for vis
    for contour in contours_final:
        temp_mask = np.zeros_like(new_mask_filtered)
        cv2.drawContours(temp_mask, [contour], 0, 255, -1)
        overlap = cv2.bitwise_and(temp_mask, mask_morph_gray)
        if np.any(overlap > 0):
            cv2.drawContours(validated_mask, [contour], 0, 255, -1)
            cv2.drawContours(validation_vis, [contour], 0, (0, 255, 0), 2) # Green outline
        else:
            cv2.drawContours(validation_vis, [contour], 0, (0, 0, 255), 2) # Red outline
    # vis_list.append(add_text(validation_vis, "6a. Validation")) # Optional validation vis
    vis_list.append(add_text(cv2.cvtColor(validated_mask.copy(), cv2.COLOR_GRAY2BGR), "6. Validated Mask"))

    # --- Create Grid ---
    # Resize all images to a common size (e.g., height of the first image / 2)
    target_h = image.shape[0] // 2
    target_w = image.shape[1] // 2 # Maintain aspect ratio roughly
    resized_vis = []
    for img in vis_list:
         # Ensure BGR
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        resized_img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        resized_vis.append(resized_img)

    # Arrange in a grid (e.g., 3x3 or adjust as needed)
    # Need 7 images: maybe 2 rows? 4+3? Pad if necessary.
    num_images = len(resized_vis)
    cols = 4 # Try 4 columns
    rows = (num_images + cols - 1) // cols

    # Pad with black images if needed
    black_img = np.zeros_like(resized_vis[0])
    while len(resized_vis) < rows * cols:
        resized_vis.append(black_img)

    # Create rows
    grid_rows = []
    for r in range(rows):
        row_imgs = resized_vis[r*cols:(r+1)*cols]
        grid_rows.append(cv2.hconcat(row_imgs))

    # Combine rows
    debug_grid = cv2.vconcat(grid_rows)

    return debug_grid

# --- End Added Functions ---


class ImageViewer(tk.Tk):
    def __init__(self, enable_debug_features=False): # Added argument
        super().__init__()
        self.title("Cell Infiltration Computer Vision Tool")
        self.geometry("800x710")

        self.image_files = []
        self.current_image_index = -1
        self.current_image_tk = None
        self.show_mask = True
        self.debug_mode = False
        self.enable_debug_features = enable_debug_features
        self.eye_icon = "show masks 👁️"
        self.eye_crossed_icon = "hide masks 🚫"
        self.debug_icon = "debug mode 🐞"
        self.debug_off_icon = "results mode ✅"

        self.gray_morph_iterations_var = tk.IntVar(value=5)
        self.white_threshold_var = tk.IntVar(value=175)

        # State Management:
        self.image_settings = {} # {image_path: {'sensitivity': s, 'white_threshold': w, 'deleted_contours': [c1, ...]}}
        self.image_display_stats = {} # {image_path: {'gray_area': ga, 'white_area': wa, 'percentage': p}}

        # Variables for current image processing state
        self.current_corrected_cv_img = None
        self.current_base_gray_mask = None # Mask calculated from parameters BEFORE deletions
        self.current_base_white_mask = None # Mask calculated from parameters BEFORE deletions
        self.original_image_dims = None # (height, width)
        self.display_image_dims = None # (width, height) of image shown in label
        self.display_scale_factor = 1.0 # Factor to convert display coords to original coords


        # --- Widgets ---

        # Top control frames
        self.btn_frame = tk.Frame(self)
        self.btn_frame.pack(side=tk.TOP, pady=5, fill=tk.X) # Pack top

        self.btn_select_folder = tk.Button(self.btn_frame, text="Select Folder", command=self.select_folder)
        self.btn_select_folder.pack(side=tk.LEFT, padx=5)

        self.btn_prev = tk.Button(self.btn_frame, text="Previous", command=self.prev_image, state=tk.DISABLED)
        self.btn_prev.pack(side=tk.LEFT, padx=5)

        self.btn_next = tk.Button(self.btn_frame, text="Next", command=self.next_image, state=tk.DISABLED)
        self.btn_next.pack(side=tk.LEFT, padx=5)

        self.btn_toggle_mask = tk.Button(self.btn_frame, text=self.eye_icon, command=self.toggle_mask, state=tk.DISABLED, width=10)
        self.btn_toggle_mask.pack(side=tk.LEFT, padx=5)
        self._update_toggle_button_text()

        # Conditionally create Debug Button
        self.btn_toggle_debug = None # Initialize to None
        if self.enable_debug_features:
            self.btn_toggle_debug = tk.Button(self.btn_frame, text=self.debug_off_icon, command=self.toggle_debug_mode, state=tk.DISABLED, width=10)
            self.btn_toggle_debug.pack(side=tk.LEFT, padx=5)
            self._update_debug_button_text() # Set initial text/icon only if button exists

        # Ensure Export Button is created
        self.btn_export = tk.Button(self.btn_frame, text="Export CSV", command=self.export_stats, state=tk.DISABLED)
        self.btn_export.pack(side=tk.LEFT, padx=5)


        self.slider_frame = tk.Frame(self)
        self.slider_frame.pack(side=tk.TOP, pady=2, fill=tk.X, padx=10) # Reduced pady

        self.lbl_sensitivity = tk.Label(self.slider_frame, text="Gray Filter Iter.:") # Shortened label
        self.lbl_sensitivity.pack(side=tk.LEFT, padx=(0, 5))

        self.scale_sensitivity = tk.Scale(
            self.slider_frame, from_=1, to=15, orient=tk.HORIZONTAL,
            variable=self.gray_morph_iterations_var, resolution=1, length=250, # Adjusted length
            command=self._on_sensitivity_change, state=tk.DISABLED
        )
        self.scale_sensitivity.pack(side=tk.LEFT, expand=True, fill=tk.X)

        # Frame for the white threshold slider
        self.white_slider_frame = tk.Frame(self)
        self.white_slider_frame.pack(side=tk.TOP, pady=2, fill=tk.X, padx=10) # Pack below sensitivity

        self.lbl_white_threshold = tk.Label(self.white_slider_frame, text="White Threshold:")
        self.lbl_white_threshold.pack(side=tk.LEFT, padx=(0, 5))

        self.scale_white_threshold = tk.Scale(
            self.white_slider_frame,
            from_=100, # Adjust range as needed
            to=254,
            orient=tk.HORIZONTAL,
            variable=self.white_threshold_var,
            resolution=1,
            length=250, # Match sensitivity slider length
            command=self._on_white_threshold_change,
            state=tk.DISABLED # Initially disabled
        )
        self.scale_white_threshold.pack(side=tk.LEFT, expand=True, fill=tk.X)


        # Bottom status frame
        self.status_frame = tk.Frame(self, bd=1, relief=tk.SUNKEN)
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X) # Pack bottom

        # Status labels inside the bottom frame
        self.lbl_status = tk.Label(self.status_frame, text="Select a folder to view images.", anchor=tk.W)
        self.lbl_status.pack(side=tk.TOP, fill=tk.X) # Pack top within status_frame

        self.lbl_total_percentage = tk.Label(self.status_frame, text="Total Infiltration: N/A", anchor=tk.W)
        self.lbl_total_percentage.pack(side=tk.TOP, fill=tk.X) # Pack top within status_frame (below lbl_status)

        # Image display label (fills the remaining space)
        self.lbl_image = tk.Label(self)
        self.lbl_image.pack(side=tk.TOP, expand=True, fill=tk.BOTH, padx=10, pady=10) # Pack last to fill space
        self.lbl_image.bind("<Button-1>", self._on_image_click) # Bind left mouse click


    def select_folder(self):
        folder_path = filedialog.askdirectory()
        if not folder_path:
            return

        self.image_files = []
        supported_formats = ('.png', '.jpg', '.jpeg', '.tif', '.tiff') # Added tiff
        try:
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(supported_formats):
                    self.image_files.append(os.path.join(folder_path, filename))
        except OSError as e:
            messagebox.showerror("Error", f"Could not read folder: {e}")
            return

        self.image_files.sort()

        # Reset state
        self.image_settings = {}
        self.image_display_stats = {}
        self.current_corrected_cv_img = None
        self.current_base_gray_mask = None
        self.current_base_white_mask = None
        self.original_image_dims = None
        self.display_image_dims = None
        self.display_scale_factor = 1.0

        if not self.image_files:
            self.current_image_index = -1
            self.lbl_image.config(image='', text="No images found in the selected folder.")
            self._update_display_info() # Update display (will show N/A)
            self.update_button_states() # Still need to disable buttons
            self.current_image_tk = None
            messagebox.showinfo("Info", "No supported image files found in the selected folder.")
        else:
            self.current_image_index = 0
            self.show_image() # Calculate stats for first image and update display
            self.update_button_states()


    def show_image(self):
        """Loads, processes, calculates stats, and displays the current image."""
        if not (0 <= self.current_image_index < len(self.image_files)):
            self.lbl_image.config(image='', text="No image selected or index out of bounds.")
            self.current_image_tk = None
            self.current_corrected_cv_img = None
            self.current_base_gray_mask = None
            self.current_base_white_mask = None
            self._update_display_info()
            return

        image_path = self.image_files[self.current_image_index]

        try:
            # --- 1. Load Image and Apply Gamma Correction ---
            img_pil = Image.open(image_path).convert('RGB')
            img_cv_rgb = np.array(img_pil)
            img_cv_bgr = cv2.cvtColor(img_cv_rgb, cv2.COLOR_RGB2BGR)
            target_mean = 100
            self.current_corrected_cv_img, _ = gamma_correct_target_d(img_cv_bgr, target_mean)
            self.original_image_dims = self.current_corrected_cv_img.shape[:2] # (height, width)

            # --- 2. Load/Initialize Settings (Parameters and Deletions) ---
            settings = self.image_settings.get(image_path, {})
            current_sensitivity = settings.get('sensitivity', self.gray_morph_iterations_var.get())
            current_white_threshold = settings.get('white_threshold', self.white_threshold_var.get())
            # Ensure deleted_contours is a list of numpy arrays
            deleted_list_raw = settings.get('deleted_contours', [])
            deleted_list = [np.array(c, dtype=np.int32) for c in deleted_list_raw] # Reconstruct numpy arrays if needed

            # Update sliders to reflect loaded/current settings
            self.gray_morph_iterations_var.set(current_sensitivity)
            self.white_threshold_var.set(current_white_threshold)

            # --- 3. Calculate Base Masks (using current parameters) ---
            self.current_base_gray_mask, self.current_base_white_mask = extract_cells_infiltrations_masks(
                self.current_corrected_cv_img,
                threshold=target_mean,
                gray_morph_iterations=current_sensitivity,
                white_threshold=current_white_threshold
            )

            # --- 4. Apply Deletions to Create Active Masks ---
            active_gray_mask = self.current_base_gray_mask.copy()
            active_white_mask = self.current_base_white_mask.copy()
            if deleted_list:
                cv2.drawContours(active_gray_mask, deleted_list, -1, 0, -1) # Fill deleted with black
                cv2.drawContours(active_white_mask, deleted_list, -1, 0, -1)

            # --- 5. Calculate Statistics (based on active masks) ---
            gray_area = np.count_nonzero(active_gray_mask)
            white_area = np.count_nonzero(active_white_mask)
            percentage = (white_area / gray_area * 100) if gray_area > 0 else 0

            # Store calculated stats for display
            self.image_display_stats[image_path] = {
                'gray_area': gray_area,
                'white_area': white_area,
                'percentage': percentage
            }

            # --- 6. Store/Update Settings (including potentially modified deleted_list) ---
            # Convert contours back to lists for storage if needed (e.g., for JSON compatibility later)
            # For now, keeping them as numpy arrays in memory is fine.
            self.image_settings[image_path] = {
                'sensitivity': current_sensitivity,
                'white_threshold': current_white_threshold,
                'deleted_contours': deleted_list # Store the list of numpy arrays
            }

            # --- 7. Generate Visualization ---
            if self.debug_mode:
                processed_img_cv = generate_debug_visualization(
                    self.current_corrected_cv_img,
                    threshold=target_mean,
                    gray_morph_iterations=current_sensitivity
                )
            else:
                if self.show_mask:
                    processed_img_cv = visualize_result(
                        self.current_corrected_cv_img,
                        self.current_base_gray_mask, # Pass base masks
                        self.current_base_white_mask,
                        deleted_list # Pass deleted contours
                    )
                else:
                    # Show only the corrected image if masks are hidden
                    processed_img_cv = self.current_corrected_cv_img

            # --- 8. Display Image ---
            processed_img_pil = Image.fromarray(cv2.cvtColor(processed_img_cv, cv2.COLOR_BGR2RGB))

            # Resize for display label, respecting aspect ratio
            lbl_w = self.lbl_image.winfo_width()
            lbl_h = self.lbl_image.winfo_height()
            if lbl_w < 50 or lbl_h < 50: lbl_w, lbl_h = 600, 500 # Default size if window not ready

            img_w, img_h = processed_img_pil.size
            scale_w = lbl_w / img_w
            scale_h = lbl_h / img_h
            self.display_scale_factor = min(scale_w, scale_h) # Use the smaller scale factor

            disp_w = int(img_w * self.display_scale_factor)
            disp_h = int(img_h * self.display_scale_factor)
            self.display_image_dims = (disp_w, disp_h)

            display_img_pil = processed_img_pil.resize((disp_w, disp_h), Image.Resampling.LANCZOS)

            self.current_image_tk = ImageTk.PhotoImage(display_img_pil)
            self.lbl_image.config(image=self.current_image_tk, text="")


        except FileNotFoundError:
             self.lbl_image.config(image='', text=f"Error: File not found\n{os.path.basename(image_path)}")
             self.current_image_tk = None
             self.current_corrected_cv_img = None
        except PIL.UnidentifiedImageError:
             self.lbl_image.config(image='', text=f"Error: Cannot identify image file\n{os.path.basename(image_path)}")
             self.current_image_tk = None
             self.current_corrected_cv_img = None
        except Exception as e:
            error_msg = f"Error processing image:\n{os.path.basename(image_path)}\n{type(e).__name__}: {e}"
            self.lbl_image.config(image='', text=error_msg)
            self.current_image_tk = None
            self.current_corrected_cv_img = None
            print(f"Error processing image {image_path}:")
            traceback.print_exc()
        finally:
            # Update status bar and total percentage regardless of success/failure
            self._update_display_info()


    def _update_display_info(self):
        """Updates the status bar and recalculates/updates the total percentage label."""
        current_perc = None
        image_path = None

        # Update Status Bar (Current Image Info)
        if self.image_files and 0 <= self.current_image_index < len(self.image_files):
            image_path = self.image_files[self.current_image_index]
            image_name = os.path.basename(image_path)
            # Get percentage from calculated stats if available
            stats = self.image_display_stats.get(image_path)
            if stats:
                current_perc = stats['percentage']
                perc_text = f"{current_perc:.2f}%"
            else:
                perc_text = "N/A" # Should be calculated by show_image
            status_text = f"Image {self.current_image_index + 1} of {len(self.image_files)}: {image_name}  |  Infiltration: {perc_text}"
            self.lbl_status.config(text=status_text)
        elif not self.image_files:
             self.lbl_status.config(text="No images found.")
        else:
             self.lbl_status.config(text="No image selected.")


        # Recalculate and Update Total Percentage Label using image_display_stats
        total_w_area = 0
        total_g_area = 0
        calculated_images_count = len(self.image_display_stats)

        for stats in self.image_display_stats.values():
            # Ensure stats exist and contain the necessary keys before adding
            if stats and 'white_area' in stats and 'gray_area' in stats:
                total_w_area += stats['white_area']
                total_g_area += stats['gray_area']
            else:
                 # If stats are missing for an image that should have them,
                 # decrement the count of effectively calculated images.
                 # This can happen if an error occurred during processing for that image.
                 # Find the key corresponding to the potentially problematic 'stats' value
                 key_for_missing_stats = None
                 for k, v in self.image_display_stats.items():
                     if v is stats:
                         key_for_missing_stats = k
                         break
                 # Only decrement if the image is actually in our list
                 if key_for_missing_stats in self.image_files:
                     calculated_images_count -=1


        if total_g_area > 0:
            total_percentage = (total_w_area / total_g_area * 100)
            # Ensure calculated_images_count doesn't exceed total files
            effective_count = min(calculated_images_count, len(self.image_files))
            total_text = f"Total Infiltration ({effective_count}/{len(self.image_files)} images): {total_percentage:.2f}%"
        elif len(self.image_files) > 0:
             total_text = f"Total Infiltration (0/{len(self.image_files)} images): N/A"
        else:
            total_text = "Total Infiltration: N/A"
        self.lbl_total_percentage.config(text=total_text)


    def _on_image_click(self, event):
        """Handles clicks on the image label to toggle deletion of contours."""
        if self.debug_mode or self.current_corrected_cv_img is None or self.current_base_gray_mask is None or self.display_scale_factor <= 0:
            # Ignore clicks in debug mode, or if image/masks aren't loaded, or scale is invalid
            return

        if not (0 <= self.current_image_index < len(self.image_files)):
            return # No valid image selected

        image_path = self.image_files[self.current_image_index]

        # --- 1. Transform Click Coordinates ---
        # event.x, event.y are relative to the label widget
        # Need to map to the original image coordinates
        orig_h, orig_w = self.original_image_dims
        disp_w, disp_h = self.display_image_dims

        # Calculate offsets if image is centered in label (assuming it might be)
        # For simplicity, assume top-left alignment for now.
        # If centering is added later, adjust click coords before scaling.
        offset_x = (self.lbl_image.winfo_width() - disp_w) // 2
        offset_y = (self.lbl_image.winfo_height() - disp_h) // 2

        # Adjust click coordinates relative to the displayed image top-left
        click_x_on_img = event.x - offset_x
        click_y_on_img = event.y - offset_y

        # Scale to original image coordinates
        orig_click_x = int(click_x_on_img / self.display_scale_factor)
        orig_click_y = int(click_y_on_img / self.display_scale_factor)

        # Check bounds
        if not (0 <= orig_click_x < orig_w and 0 <= orig_click_y < orig_h):
            # print("Click outside image bounds") # Debug
            return

        click_point = (orig_click_x, orig_click_y)

        # --- 2. Load Current Settings and Deleted List ---
        settings = self.image_settings.get(image_path, {})
        deleted_list_raw = settings.get('deleted_contours', [])
        # Make sure we have a list of numpy arrays to work with
        deleted_list = [np.array(c, dtype=np.int32) for c in deleted_list_raw]
        modified = False
        contour_to_modify = None
        is_reinstating = False

        # --- 3. Check if Clicking on a Deleted Contour (to reinstate) ---
        found_deleted_index = -1
        for i, contour in enumerate(deleted_list):
            # pointPolygonTest returns +1 (inside), -1 (outside), 0 (on contour)
            distance = cv2.pointPolygonTest(contour, click_point, False)
            if distance >= 0: # Click is inside or on the contour
                found_deleted_index = i
                is_reinstating = True
                break

        if is_reinstating:
            contour_to_modify = deleted_list.pop(found_deleted_index)
            modified = True
            print(f"Reinstating contour {found_deleted_index}") # Debug
        else:
            # --- 4. Check if Clicking on an Active Contour (to delete) ---
            # Find contours on the *base* mask (before deletions)
            base_contours, _ = cv2.findContours(self.current_base_gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            found_active_contour = None
            for contour in base_contours:
                distance = cv2.pointPolygonTest(contour, click_point, False)
                if distance >= 0:
                    # Check if this contour is *already* in the deleted list
                    is_already_deleted = False
                    for deleted_c in deleted_list:
                        # Simple comparison (might need more robust check if points can reorder)
                        if np.array_equal(contour, deleted_c):
                            is_already_deleted = True
                            break
                    if not is_already_deleted:
                        found_active_contour = contour
                        break # Found the active contour to delete

            if found_active_contour is not None:
                deleted_list.append(found_active_contour)
                modified = True
                print(f"Deleting new contour") # Debug


        # --- 5. Update State and Refresh if Modified ---
        if modified:
            # Update the settings dictionary
            settings['deleted_contours'] = deleted_list
            self.image_settings[image_path] = settings
            # Refresh the display (recalculates stats and redraws)
            self.show_image()


    def _on_sensitivity_change(self, value):
        """Callback function when the sensitivity slider value changes."""
        if not (0 <= self.current_image_index < len(self.image_files)): return
        image_path = self.image_files[self.current_image_index]
        settings = self.image_settings.get(image_path, {})
        new_sensitivity = self.gray_morph_iterations_var.get()

        # Only update and refresh if the value actually changed
        if settings.get('sensitivity') != new_sensitivity:
            print(f"Sensitivity changed for {os.path.basename(image_path)} to {new_sensitivity}, refreshing...")
            settings['sensitivity'] = new_sensitivity
            # Keep existing deleted contours
            settings['deleted_contours'] = settings.get('deleted_contours', [])
            self.image_settings[image_path] = settings
            self.show_image() # Refresh view, recalculate masks/stats

    def _on_white_threshold_change(self, value):
        """Callback function when the white threshold slider value changes."""
        if not (0 <= self.current_image_index < len(self.image_files)): return
        image_path = self.image_files[self.current_image_index]
        settings = self.image_settings.get(image_path, {})
        new_threshold = self.white_threshold_var.get()

        # Only update and refresh if the value actually changed
        if settings.get('white_threshold') != new_threshold:
            print(f"White threshold changed for {os.path.basename(image_path)} to {new_threshold}, refreshing...")
            settings['white_threshold'] = new_threshold
            # Keep existing deleted contours
            settings['deleted_contours'] = settings.get('deleted_contours', [])
            self.image_settings[image_path] = settings
            self.show_image() # Refresh view, recalculate masks/stats


    def toggle_mask(self):
        """Toggles the visibility of the mask overlay."""
        if self.debug_mode: # Turn off debug mode if mask toggle is used
            self.debug_mode = False
            self._update_debug_button_text()
        self.show_mask = not self.show_mask
        self._update_toggle_button_text()
        self.show_image() # Refresh the image display

    def toggle_debug_mode(self):
        """Toggles the debug visualization grid."""
        self.debug_mode = not self.debug_mode
        if self.debug_mode: # If turning debug on, ensure mask overlay is conceptually 'off'
            self.show_mask = False
            self._update_toggle_button_text()
        self._update_debug_button_text()
        self.show_image() # Refresh the image display


    def _update_toggle_button_text(self):
        """Updates the text/icon of the toggle mask button."""
        if self.show_mask and not self.debug_mode: # Only show eye if mask is on AND debug is off
            self.btn_toggle_mask.config(text=self.eye_icon)
        else:
            self.btn_toggle_mask.config(text=self.eye_crossed_icon)

    def _update_debug_button_text(self):
        """Updates the text/icon of the toggle debug button if it exists."""
        if self.btn_toggle_debug: # Check if button exists
            if self.debug_mode:
                self.btn_toggle_debug.config(text=self.debug_icon)
            else:
                self.btn_toggle_debug.config(text=self.debug_off_icon)


    def next_image(self):
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.show_image() # Handles loading/calculating stats and updating display
            self.update_button_states()

    def prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_image() # Handles loading/calculating stats and updating display
            self.update_button_states()

    def export_stats(self):
        """Exports the calculated statistics to a CSV file."""
        if not self.image_display_stats:
            messagebox.showinfo("Export Stats", "No statistics calculated yet. Please process some images.")
            return

        # Ask user for save location
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save Statistics As"
        )
        if not filepath: return

        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)

                # Write header
                header = ['Image Filename', 'Gray Area (pixels)', 'White Area (pixels)', 'Infiltration (%)', 'Sensitivity (Gray Iter)', 'White Threshold']
                writer.writerow(header)

                # Write image data using image_display_stats and image_settings
                total_w_area = 0
                total_g_area = 0
                # Iterate through display stats which reflect current calculations
                for image_path, stats in self.image_display_stats.items():
                    filename = os.path.basename(image_path)
                    # Get corresponding settings for parameters
                    settings = self.image_settings.get(image_path, {})
                    sensitivity = settings.get('sensitivity', 'N/A')
                    white_threshold = settings.get('white_threshold', 'N/A')

                    # Ensure stats dictionary is not None and contains keys
                    gray_area_val = stats.get('gray_area', 0) if stats else 0
                    white_area_val = stats.get('white_area', 0) if stats else 0
                    percentage_val = stats.get('percentage', 0) if stats else 0

                    row = [
                        filename,
                        gray_area_val,
                        white_area_val,
                        f"{percentage_val:.2f}", # Format percentage
                        sensitivity,
                        white_threshold
                    ]
                    writer.writerow(row)
                    total_w_area += white_area_val
                    total_g_area += gray_area_val

                # Write total row (calculation remains the same)
                if total_g_area > 0:
                    total_percentage = (total_w_area / total_g_area * 100)
                    total_perc_str = f"{total_percentage:.2f}"
                else:
                    total_perc_str = "N/A"

                writer.writerow([]) # Add an empty row for separation
                writer.writerow(['TOTAL', total_g_area, total_w_area, total_perc_str, '', ''])

            messagebox.showinfo("Export Successful", f"Statistics successfully exported to:\n{filepath}")

        except IOError as e:
            messagebox.showerror("Export Error", f"Could not write file:\n{e}")
        except Exception as e:
            messagebox.showerror("Export Error", f"An unexpected error occurred during export:\n{e}")
            traceback.print_exc()


    def update_button_states(self):
        if not self.image_files:
            self.btn_prev.config(state=tk.DISABLED)
            self.btn_next.config(state=tk.DISABLED)
            self.btn_toggle_mask.config(state=tk.DISABLED)
            if self.btn_toggle_debug:
                self.btn_toggle_debug.config(state=tk.DISABLED)
            self.scale_sensitivity.config(state=tk.DISABLED)
            self.scale_white_threshold.config(state=tk.DISABLED) # Disable white threshold slider
            self.btn_export.config(state=tk.DISABLED) # Disable export button
            return

        # Enable buttons and sliders once images are loaded
        self.btn_toggle_mask.config(state=tk.NORMAL)
        if self.btn_toggle_debug:
            self.btn_toggle_debug.config(state=tk.NORMAL)
            self._update_debug_button_text() # Ensure text is correct only if button exists
        self.scale_sensitivity.config(state=tk.NORMAL)
        self.scale_white_threshold.config(state=tk.NORMAL) # Enable white threshold slider
        self._update_toggle_button_text()


        # Enable export button only if display stats exist
        if self.image_display_stats:
             self.btn_export.config(state=tk.NORMAL)
        else:
             self.btn_export.config(state=tk.DISABLED)


        if self.current_image_index <= 0:
            self.btn_prev.config(state=tk.DISABLED)
        else:
            self.btn_prev.config(state=tk.NORMAL)

        if self.current_image_index >= len(self.image_files) - 1:
            self.btn_next.config(state=tk.DISABLED)
        else:
            self.btn_next.config(state=tk.NORMAL)

    def _on_resize(self, event):
        # Debounce or delay resize handling if needed
        # For now, redraw if the widget that resized is the image label itself
        # and we have an image loaded.
        if event.widget == self.lbl_image and self.current_image_tk:
             # Check if size actually changed significantly to avoid rapid redraws
             # This simple check might not be perfect
             new_w, new_h = event.width, event.height
             if self.display_image_dims and (abs(new_w - self.display_image_dims[0]) > 5 or abs(new_h - self.display_image_dims[1]) > 5):
                 self.show_image() # Re-render image on resize to fit

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Cell Infiltration Computer Vision Tool")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug visualization features.")
    args = parser.parse_args()

    # Pass the debug flag to the application
    app = ImageViewer(enable_debug_features=args.debug)
    # Bind resize event to the main window, _on_resize checks the widget
    app.bind('<Configure>', app._on_resize)
    app.mainloop()