#!/usr/bin/env python3
"""
Easy launcher for the Real-time Face Mask Detection Demo
Shows compatibility with any model from yewon_pipeline
"""

import os
import sys
import subprocess
import glob


def find_models():
    """Find available models in the project"""
    models = {
        'classifiers': [],
        'detectors': ['haar', 'yunet']
    }

    # Search for .pth files relative to yewon_pipeline directory
    search_paths = [
        'runs_12k/**/*.pth',
        'runs_eval/**/*.pth',
        '../model_artifacts/*.pth',
        '*.pth'
    ]

    for pattern in search_paths:
        for path in glob.glob(pattern, recursive=True):
            models['classifiers'].append({
                'name': os.path.basename(path),
                'path': path
            })

    return models


def print_menu():
    """Print interactive menu"""
    models = find_models()

    print("\n" + "="*70)
    print("REAL-TIME FACE MASK DETECTION - MODEL COMPATIBILITY DEMO")
    print("="*70)
    print("\nThis demo showcases the flexibility of the system to work with")
    print("any detector and classifier combination from yewon_pipeline")
    print("\n" + "="*70)

    print("\nAVAILABLE CONFIGURATIONS:")
    print("\n1. Detection Only (Haar Cascade)")
    print("   - Fast, CPU-friendly face detection")
    print("   - No mask classification")

    print("\n2. Detection Only (YuNet)")
    print("   - More accurate face detection")
    print("   - No mask classification")

    print("\n3. Detection Only (MTCNN)")
    print("   - Precise multi-stage CNN detection")
    print("   - GPU accelerated")

    print("\n4. Detection Only (RetinaFace)")
    print("   - State-of-the-art face detection")
    print("   - Best accuracy, GPU required")

    if models['classifiers']:
        print("\n5. Haar + First Available Classifier")
        print(f"   - Classifier: {models['classifiers'][0]['name']}")

        if len(models['classifiers']) > 1:
            print("\n6. YuNet + Second Available Classifier")
            print(f"   - Classifier: {models['classifiers'][1]['name']}")

    print("\n7. Interactive Mode (Switch models at runtime)")
    print("   - Start with basic setup")
    print("   - Use keyboard to switch between models")

    print("\n8. Custom Configuration")
    print("   - Specify your own detector and classifier")

    print("\n0. Exit")

    return models


def run_demo(detector='haar', classifier=None):
    """Run the demo with specified configuration"""
    cmd = ['python', 'realtime_demo.py', '--detector', detector]

    if classifier:
        cmd.extend(['--classifier', classifier])

    print(f"\nRunning: {' '.join(cmd)}")
    print("="*70)

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nDemo interrupted")


def main():
    """Main menu loop"""
    while True:
        models = print_menu()

        try:
            choice = input("\nSelect option (0-8): ").strip()

            if choice == '0':
                print("Goodbye!")
                break

            elif choice == '1':
                print("\n" + "="*70)
                print("DETECTION ONLY MODE - HAAR CASCADE")
                print("="*70)
                print("This mode demonstrates pure face detection without classification")
                run_demo('haar', None)

            elif choice == '2':
                print("\n" + "="*70)
                print("DETECTION ONLY MODE - YUNET")
                print("="*70)
                print("This mode uses YuNet for more accurate face detection")
                run_demo('yunet', None)

            elif choice == '3':
                print("\n" + "="*70)
                print("DETECTION ONLY MODE - MTCNN")
                print("="*70)
                print("Multi-stage CNN detection with GPU acceleration")
                run_demo('mtcnn', None)

            elif choice == '4':
                print("\n" + "="*70)
                print("DETECTION ONLY MODE - RETINAFACE")
                print("="*70)
                print("State-of-the-art face detection")
                run_demo('retinaface', None)

            elif choice == '5' and models['classifiers']:
                classifier_path = models['classifiers'][0]['path']
                print("\n" + "="*70)
                print("HAAR DETECTOR + CLASSIFIER")
                print(f"Using: {models['classifiers'][0]['name']}")
                print("="*70)
                run_demo('haar', classifier_path)

            elif choice == '6' and len(models['classifiers']) > 1:
                classifier_path = models['classifiers'][1]['path']
                print("\n" + "="*70)
                print("YUNET DETECTOR + CLASSIFIER")
                print(f"Using: {models['classifiers'][1]['name']}")
                print("="*70)
                run_demo('yunet', classifier_path)

            elif choice == '7':
                print("\n" + "="*70)
                print("INTERACTIVE MODE")
                print("="*70)
                print("\nIn this mode, you can switch between models at runtime:")
                print("\nKEYBOARD CONTROLS:")
                print("  Detectors:")
                print("    1: Haar Cascade (fast, CPU)")
                print("    2: YuNet (accurate, ONNX)")
                print("    3: MTCNN (precise, GPU)")
                print("    4: RetinaFace (best quality, GPU)")
                print("  Classifiers:")
                print("    5/6/7: Load available models")
                print("    0: Disable classifier")
                print("  Display:")
                print("    I: Toggle info | F: Toggle FPS")
                print("    S: Save screenshot | Q: Quit")
                print("\nThis demonstrates the system's flexibility to switch")
                print("between any detector/classifier combination in real-time!")
                input("\nPress Enter to start...")
                run_demo('haar', None)

            elif choice == '8':
                print("\n" + "="*70)
                print("CUSTOM CONFIGURATION")
                print("="*70)

                print("\nAvailable detectors: haar, yunet, mtcnn, retinaface")
                detector = input("Enter detector type (default: haar): ").strip() or 'haar'

                print("\nAvailable classifiers:")
                if models['classifiers']:
                    for i, model in enumerate(models['classifiers'], 1):
                        print(f"  {i}. {model['path']}")
                    print("  0. No classifier (detection only)")

                    idx = input("Select classifier (0 for none): ").strip()
                    if idx and idx != '0':
                        try:
                            idx = int(idx) - 1
                            if 0 <= idx < len(models['classifiers']):
                                classifier = models['classifiers'][idx]['path']
                            else:
                                classifier = None
                        except:
                            classifier = None
                    else:
                        classifier = None
                else:
                    print("  No classifiers found")
                    classifier = None

                run_demo(detector, classifier)

            else:
                print("Invalid choice or option not available")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        except Exception as e:
            print(f"Error: {e}")

        print("\n" + "="*70)
        input("Press Enter to continue...")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("WELCOME TO THE UNIFIED FACE MASK DETECTION SYSTEM")
    print("="*70)
    print("\nThis system demonstrates compatibility with ANY model combination")
    print("from the yewon_pipeline, including:")
    print("  - Different face detectors (Haar, YuNet)")
    print("  - Any trained classifier model (PyTorch, Custom CNN, etc.)")
    print("  - Real-time model switching")
    print("  - Flexible input processing")

    main()