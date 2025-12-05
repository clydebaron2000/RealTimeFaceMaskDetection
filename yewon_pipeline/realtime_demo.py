#!/usr/bin/env python3
"""
Real-time Face Mask Detection Demo
Demonstrates compatibility with any model from yewon_pipeline
"""

import cv2
import torch
import numpy as np
import os
import sys
import time
import argparse
from typing import Optional, List, Dict

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from unified_model_loader import UnifiedPipeline, UnifiedModelLoader


class RealtimeDemo:
    """
    Real-time demonstration system with dynamic model switching
    """

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None
        self.is_running = False

        # Model loader for discovering available models
        self.model_loader = UnifiedModelLoader()
        self.available_models = self.model_loader.list_available_models()

        # Current pipeline
        self.pipeline = UnifiedPipeline(detector_type='haar')

        # UI state
        self.show_info = True
        self.show_fps = True
        self.current_detector_idx = 0
        self.current_classifier_idx = -1  # -1 means no classifier

        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

        # Initialize camera
        self._initialize_camera()

    def _initialize_camera(self):
        """Initialize webcam"""
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

    def _update_fps(self):
        """Update FPS calculation"""
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            current_time = time.time()
            self.fps = 30 / (current_time - self.start_time)
            self.start_time = current_time

    def _draw_ui(self, frame: np.ndarray) -> np.ndarray:
        """Draw UI elements on frame"""
        h, w = frame.shape[:2]

        # Draw FPS
        if self.show_fps:
            fps_text = f"FPS: {self.fps:.1f}"
            cv2.putText(frame, fps_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw model info
        if self.show_info:
            # Current detector
            detector_text = f"Detector: {self.pipeline.detector.detector_type}"
            cv2.putText(frame, detector_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Current classifier
            if self.pipeline.classifier:
                if self.pipeline.classifier_info:
                    classifier_name = self.pipeline.classifier_info.get('name', 'Unknown')
                else:
                    classifier_name = "Loaded"
                classifier_text = f"Classifier: {classifier_name}"
            else:
                classifier_text = "Classifier: None (detection only)"

            cv2.putText(frame, classifier_text, (10, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Draw controls
            controls = [
                "Controls:",
                "1: Haar | 2: YuNet | 3: MTCNN | 4: RetinaFace",
                "5/6/7: Load classifier models",
                "0: Disable classifier",
                "I: Toggle info display",
                "F: Toggle FPS",
                "S: Save screenshot",
                "Q: Quit"
            ]

            y_offset = h - len(controls) * 20 - 10
            for i, text in enumerate(controls):
                cv2.putText(frame, text, (10, y_offset + i*20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return frame

    def _handle_key(self, key: int) -> bool:
        """Handle keyboard input"""
        if key == ord('q') or key == ord('Q'):
            return False

        elif key == ord('1'):
            # Switch to Haar detector
            self.pipeline.detector = FlexibleDetector('haar', device=self.pipeline.device)
            print("Switched to Haar detector")

        elif key == ord('2'):
            # Switch to YuNet detector
            self.pipeline.detector = FlexibleDetector('yunet', device=self.pipeline.device)
            print("Switched to YuNet detector")

        elif key == ord('3'):
            # Switch to MTCNN detector
            self.pipeline.detector = FlexibleDetector('mtcnn', device=self.pipeline.device)
            print("Switched to MTCNN detector")

        elif key == ord('4'):
            # Switch to RetinaFace detector
            self.pipeline.detector = FlexibleDetector('retinaface', device=self.pipeline.device)
            print("Switched to RetinaFace detector")

        elif key == ord('5'):
            # Load first available classifier
            if self.available_models['classifiers']:
                model_info = self.available_models['classifiers'][0]
                self.pipeline.load_classifier(model_info['path'])
                print(f"Loaded classifier: {model_info['name']}")

        elif key == ord('6'):
            # Load second available classifier
            if len(self.available_models['classifiers']) > 1:
                model_info = self.available_models['classifiers'][1]
                self.pipeline.load_classifier(model_info['path'])
                print(f"Loaded classifier: {model_info['name']}")

        elif key == ord('7'):
            # Load third available classifier
            if len(self.available_models['classifiers']) > 2:
                model_info = self.available_models['classifiers'][2]
                self.pipeline.load_classifier(model_info['path'])
                print(f"Loaded classifier: {model_info['name']}")

        elif key == ord('0'):
            # Disable classifier
            self.pipeline.classifier = None
            self.pipeline.classifier_info = None
            print("Classifier disabled - detection only")

        elif key == ord('i') or key == ord('I'):
            self.show_info = not self.show_info

        elif key == ord('f') or key == ord('F'):
            self.show_fps = not self.show_fps

        elif key == ord('s') or key == ord('S'):
            # Save screenshot
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, self.last_frame)
            print(f"Screenshot saved: {filename}")

        return True

    def print_available_models(self):
        """Print all available models"""
        print("\n" + "="*60)
        print("AVAILABLE MODELS")
        print("="*60)

        print("\nDetectors:")
        for detector in self.available_models['detectors']:
            print(f"  - {detector['name']} ({detector['type']})")

        print("\nClassifiers:")
        if self.available_models['classifiers']:
            for i, classifier in enumerate(self.available_models['classifiers'], 1):
                print(f"  {i}. {classifier['name']}")
                print(f"     Path: {classifier['path']}")
                print(f"     Classes: {classifier['num_classes']}")
                print(f"     Input size: {classifier['img_size']}")
        else:
            print("  No classifier models found")

        print("\n" + "="*60)

    def run(self):
        """Main demo loop"""
        print("\n" + "="*60)
        print("REAL-TIME FACE MASK DETECTION DEMO")
        print("Compatible with any model from yewon_pipeline")
        print("="*60)

        self.print_available_models()

        print("\nStarting webcam feed...")
        print("\nKeyboard shortcuts:")
        print("  Detectors: 1=Haar, 2=YuNet, 3=MTCNN, 4=RetinaFace")
        print("  Classifiers: 5/6/7=Load models, 0=Disable")
        print("  Press 'Q' to quit\n")

        self.is_running = True

        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    break

                # Store for screenshot
                self.last_frame = frame.copy()

                # Process frame
                results = self.pipeline.process_frame(frame)

                # Draw results
                frame = self.pipeline.draw_results(frame, results)

                # Update FPS
                self._update_fps()

                # Draw UI
                frame = self._draw_ui(frame)

                # Display
                cv2.imshow("Real-time Face Mask Detection", frame)

                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_key(key):
                    break

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Demo ended")


def main():
    parser = argparse.ArgumentParser(description='Real-time Face Mask Detection Demo')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index (default: 0)')
    parser.add_argument('--detector', type=str, default='haar',
                       choices=['haar', 'yunet', 'mtcnn', 'retinaface'],
                       help='Initial detector (default: haar)')
    parser.add_argument('--classifier', type=str, default=None,
                       help='Path to classifier model (optional)')

    args = parser.parse_args()

    # Create and run demo
    demo = RealtimeDemo(camera_index=args.camera)

    # Load initial detector
    demo.pipeline.detector.detector_type = args.detector

    # Load initial classifier if specified
    if args.classifier and os.path.exists(args.classifier):
        demo.pipeline.load_classifier(args.classifier)

    # Run demo
    demo.run()


if __name__ == '__main__':
    # Import FlexibleDetector here to avoid circular import
    from unified_model_loader import FlexibleDetector
    main()