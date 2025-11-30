"""
Display manager for dual filter system
Handles side-by-side video rendering and UI elements
"""

import cv2
import numpy as np
import time
from typing import Tuple, Optional, Dict, Any


class FilterDisplay:
    """
    Manages the display of dual filter outputs side by side
    """
    
    def __init__(self, 
                 window_name: str = "Dual Filter Face Detection",
                 frame_width: int = 640,
                 frame_height: int = 480,
                 show_fps: bool = True,
                 show_stats: bool = True):
        """
        Initialize filter display manager
        
        Args:
            window_name: Name of the display window
            frame_width: Width of each filter frame
            frame_height: Height of each filter frame
            show_fps: Whether to show FPS counter
            show_stats: Whether to show statistics
        """
        self.window_name = window_name
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.show_fps = show_fps
        self.show_stats = show_stats
        
        # Calculate combined display dimensions
        self.display_width = frame_width * 2 + 20  # 20px gap between frames
        self.display_height = frame_height + 120   # More space for UI elements below frames
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = None
        self.fps = 0.0
        self.last_fps_update = 0
        
        # UI colors
        self.ui_colors = {
            'background': (40, 40, 40),
            'text': (255, 255, 255),
            'accent': (0, 255, 255),
            'separator': (100, 100, 100)
        }
        
        # Initialize window
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
    
    def create_combined_display(self, 
                              left_frame: np.ndarray, 
                              right_frame: np.ndarray,
                              left_stats: Optional[Dict[str, Any]] = None,
                              right_stats: Optional[Dict[str, Any]] = None,
                              detector_info: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Create combined side-by-side display
        
        Args:
            left_frame: Left filter output frame
            right_frame: Right filter output frame
            left_stats: Statistics for left filter
            right_stats: Statistics for right filter
            
        Returns:
            Combined display frame
        """
        # Resize frames to standard size
        left_resized = cv2.resize(left_frame, (self.frame_width, self.frame_height))
        right_resized = cv2.resize(right_frame, (self.frame_width, self.frame_height))
        
        # Create combined frame with background
        combined = np.full(
            (self.display_height, self.display_width, 3),
            self.ui_colors['background'],
            dtype=np.uint8
        )
        
        # Place left frame
        combined[50:50+self.frame_height, 0:self.frame_width] = left_resized
        
        # Place right frame
        right_x = self.frame_width + 20
        combined[50:50+self.frame_height, right_x:right_x+self.frame_width] = right_resized
        
        # Add separator line
        separator_x = self.frame_width + 10
        cv2.line(
            combined,
            (separator_x, 50),
            (separator_x, 50 + self.frame_height),
            self.ui_colors['separator'],
            2
        )
        
        # Add titles
        self._add_filter_titles(combined, left_stats, right_stats, detector_info)
        
        # Add global UI elements
        if self.show_fps:
            self._add_fps_counter(combined)
        
        if self.show_stats:
            self._add_global_stats(combined, left_stats, right_stats)
        
        # Add detector/classifier lists and threshold
        if detector_info:
            self._add_detector_classifier_info(combined, detector_info)
            self._add_threshold_info(combined, detector_info)
        
        # Add controls help
        self._add_controls_help(combined)
        
        return combined
    
    def _add_filter_titles(self, 
                          frame: np.ndarray, 
                          left_stats: Optional[Dict[str, Any]], 
                          right_stats: Optional[Dict[str, Any]],
                          detector_info: Optional[Dict[str, Any]] = None):
        """Add titles for each filter with dynamic detector/classifier names"""
        # Get current detector and classifier names from detector_info
        current_detector = "Detection"
        current_classifier = "Classification"
        
        if detector_info:
            available_detectors = detector_info.get('available_detectors', [])
            available_classifiers = detector_info.get('available_classifiers', [])
            current_detector_index = detector_info.get('current_detector_index', 0)
            current_classifier_index = detector_info.get('current_classifier_index', 0)
            
            if current_detector_index < len(available_detectors):
                current_detector = available_detectors[current_detector_index].upper()
            
            if current_classifier_index < len(available_classifiers):
                current_classifier = available_classifiers[current_classifier_index].upper()
        
        # Left filter title - shows current detector
        left_title = f"Detection: {current_detector}"
        cv2.putText(
            frame,
            left_title,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            self.ui_colors['accent'],
            2
        )
        
        # Right filter title - shows current detector + classifier
        right_title = f"Classification: {current_detector} + {current_classifier}"
        right_x = self.frame_width + 30
        cv2.putText(
            frame,
            right_title,
            (right_x, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            self.ui_colors['accent'],
            2
        )
    
    def _add_fps_counter(self, frame: np.ndarray):
        """Add FPS counter to display"""
        # Update FPS calculation
        current_time = time.time()
        if self.start_time is None:
            self.start_time = current_time
        
        self.frame_count += 1
        elapsed = current_time - self.start_time
        
        # Update FPS every second
        if current_time - self.last_fps_update >= 1.0:
            if elapsed > 0:
                self.fps = self.frame_count / elapsed
            self.last_fps_update = current_time
        
        # Display FPS
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(
            frame,
            fps_text,
            (self.display_width - 120, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            self.ui_colors['text'],
            2
        )
    
    def _add_global_stats(self, 
                         frame: np.ndarray, 
                         left_stats: Optional[Dict[str, Any]], 
                         right_stats: Optional[Dict[str, Any]]):
        """Add global statistics comparison - positioned lower and out of picture frames"""
        # Position stats below the image frames (out of picture frame area)
        stats_y = 50 + self.frame_height + 40  # Below the frames with some padding
        
        # Left filter stats - positioned lower
        if left_stats:
            left_text = f"L: {left_stats.get('current_detections', 0)} faces"
            cv2.putText(
                frame,
                left_text,
                (10, stats_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.ui_colors['text'],
                2
            )
        
        # Right filter stats - positioned lower with more room
        if right_stats:
            detections = right_stats.get('current_detections', 0)
            classifications = right_stats.get('current_classifications', 0)
            right_text = f"R: {detections} faces, {classifications} classified"
            
            right_x = self.frame_width + 30
            cv2.putText(
                frame,
                right_text,
                (right_x, stats_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.ui_colors['text'],
                2
            )
    
    def _add_detector_classifier_info(self, frame: np.ndarray, detector_info: Dict[str, Any]):
        """Add detector/classifier lists and threshold info to the display"""
        info_y_start = self.display_height - 50
        
        # Get detector and classifier info
        available_detectors = detector_info.get('available_detectors', [])
        available_classifiers = detector_info.get('available_classifiers', [])
        current_detector_index = detector_info.get('current_detector_index', 0)
        current_classifier_index = detector_info.get('current_classifier_index', 0)
        confidence_threshold = detector_info.get('confidence_threshold', 0.6)
        
        # Display detector list (left side)
        detector_text = "Detectors: "
        for i, detector in enumerate(available_detectors):
            if i == current_detector_index:
                detector_text += f"[{detector.upper()}] "
            else:
                detector_text += f"{detector} "
        
        cv2.putText(
            frame,
            detector_text,
            (10, info_y_start),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.ui_colors['text'],
            1
        )
        
        # Display classifier list (right side)
        classifier_text = "Classifiers: "
        for i, classifier in enumerate(available_classifiers):
            if i == current_classifier_index:
                classifier_text += f"[{classifier.upper()}] "
            else:
                classifier_text += f"{classifier} "
        
        right_x = self.frame_width + 30
        cv2.putText(
            frame,
            classifier_text,
            (right_x, info_y_start),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.ui_colors['text'],
            1
        )
    
    def _add_threshold_info(self, frame: np.ndarray, detector_info: Dict[str, Any]):
        """Add threshold info positioned on top right of left header for detection"""
        if not detector_info:
            return
        
        confidence_threshold = detector_info.get('confidence_threshold', 0.6)
        threshold_text = f"Threshold: {confidence_threshold:.2f}"
        
        # Calculate text width to position it properly on the right side of left header
        (text_width, text_height), _ = cv2.getTextSize(
            threshold_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Position threshold on top right of left side header area
        # Left frame area goes from x=0 to x=frame_width, so position near the right edge
        threshold_x = self.frame_width - text_width - 10  # 10px padding from right edge of left frame
        threshold_y = 30  # Same level as the header titles
        
        cv2.putText(
            frame,
            threshold_text,
            (threshold_x, threshold_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.ui_colors['accent'],
            1
        )
    
    def _add_controls_help(self, frame: np.ndarray):
        """Add keyboard controls help"""
        help_y = self.display_height - 15
        help_text = "Controls: Q=Quit | S=Screenshot | D=Cycle Detector | C=Cycle Classifier"
        
        # Calculate text size to center it
        (text_width, text_height), _ = cv2.getTextSize(
            help_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        text_x = (self.display_width - text_width) // 2
        
        cv2.putText(
            frame,
            help_text,
            (text_x, help_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.ui_colors['text'],
            1
        )
    
    def show_frame(self, combined_frame: np.ndarray) -> int:
        """
        Display the combined frame and handle input
        
        Args:
            combined_frame: Combined display frame
            
        Returns:
            Key code pressed (or -1 if no key)
        """
        cv2.imshow(self.window_name, combined_frame)
        return cv2.waitKey(1) & 0xFF
    
    def save_screenshot(self, combined_frame: np.ndarray, filename: Optional[str] = None) -> str:
        """
        Save screenshot of current display
        
        Args:
            combined_frame: Frame to save
            filename: Optional filename (auto-generated if None)
            
        Returns:
            Path to saved screenshot
        """
        if filename is None:
            timestamp = int(time.time())
            filename = f"dual_filter_screenshot_{timestamp}.jpg"
        
        cv2.imwrite(filename, combined_frame)
        return filename
    
    def create_comparison_overlay(self, 
                                left_frame: np.ndarray, 
                                right_frame: np.ndarray,
                                alpha: float = 0.5) -> np.ndarray:
        """
        Create overlay comparison of both filters
        
        Args:
            left_frame: Left filter frame
            right_frame: Right filter frame
            alpha: Blend factor (0.0 to 1.0)
            
        Returns:
            Blended comparison frame
        """
        # Resize frames to same size
        left_resized = cv2.resize(left_frame, (self.frame_width, self.frame_height))
        right_resized = cv2.resize(right_frame, (self.frame_width, self.frame_height))
        
        # Create blended overlay
        blended = cv2.addWeighted(left_resized, alpha, right_resized, 1 - alpha, 0)
        
        # Add overlay info
        cv2.putText(
            blended,
            f"Overlay Comparison (Alpha: {alpha:.1f})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        cv2.putText(
            blended,
            "Detection Only + Classification",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        return blended
    
    def create_difference_view(self, 
                             left_frame: np.ndarray, 
                             right_frame: np.ndarray) -> np.ndarray:
        """
        Create difference visualization between filters
        
        Args:
            left_frame: Left filter frame
            right_frame: Right filter frame
            
        Returns:
            Difference visualization frame
        """
        # Resize frames to same size
        left_resized = cv2.resize(left_frame, (self.frame_width, self.frame_height))
        right_resized = cv2.resize(right_frame, (self.frame_width, self.frame_height))
        
        # Convert to grayscale for difference calculation
        left_gray = cv2.cvtColor(left_resized, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_resized, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        diff = cv2.absdiff(left_gray, right_gray)
        
        # Apply threshold to highlight significant differences
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # Convert back to BGR for display
        diff_colored = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        
        # Add info
        cv2.putText(
            diff_colored,
            "Difference View",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        cv2.putText(
            diff_colored,
            "White areas show differences",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        return diff_colored
    
    def get_display_stats(self) -> Dict[str, Any]:
        """Get display statistics"""
        return {
            'fps': self.fps,
            'frame_count': self.frame_count,
            'display_width': self.display_width,
            'display_height': self.display_height,
            'uptime': time.time() - self.start_time if self.start_time else 0
        }
    
    def close(self):
        """Close display window and cleanup"""
        cv2.destroyWindow(self.window_name)
        cv2.destroyAllWindows()
