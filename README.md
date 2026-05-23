# Interaction Behavior-Based Cognitive Load Estimation

## Authors
- Fatima Noorulain (BSSE23003)
- Janasheen Haider (BSSE23110)

## Abstract
This project presents a non-intrusive, privacy-preserving multimodal Human-Computer Interaction (HCI) system for real-time cognitive load estimation using mouse and keyboard interaction behavior. Unlike traditional approaches relying on physiological sensors such as EEG or eye-trackers, this system estimates cognitive load using ordinary interaction patterns collected from a standard computer setup.

The system extracts 15 behavioral features including:
- Typing rate
- Inter-key interval statistics
- Mouse movement speed
- Pause frequency
- Backspace count

A MediaPipe face mesh webcam display is integrated to provide live visual feedback alongside predicted cognitive load levels.

## Features
- Real-time mouse and keyboard event logging
- 15 behavioral feature extraction pipeline
- Machine learning–based cognitive load classification
- MediaPipe 478-point face mesh visualization
- Live webcam HUD display
- Random Forest, SVM, k-NN, and Gradient Boosting evaluation

## Machine Learning Models
The following classifiers were trained and evaluated:
- Random Forest
- Support Vector Machine (SVM)
- k-Nearest Neighbors (k-NN)
- Gradient Boosting

Random Forest achieved the highest performance:
- Test Accuracy: 93.9%
- Cross-Validated Accuracy: 93.0% (±10.2%)

## Technologies Used
- Python 3.12
- Scikit-learn
- OpenCV
- MediaPipe
- pynput
- NumPy
- Pandas

## System Pipeline
Input → Logging → Feature Extraction → Classification → Live Display

1. Mouse and keyboard events are logged in real time
2. Behavioral features are extracted
3. ML classifier predicts cognitive load:
   - LOW
   - MEDIUM
   - HIGH
4. Prediction is displayed on webcam feed with MediaPipe overlay

## Dataset
The dataset contains:
- 27 real participant sessions
- 300 synthetic sessions generated using statistics inspired by the Aalto 136M Keystrokes dataset

Final dataset size:
- 327 sessions
- Balanced across Low / Medium / High classes

## Project Structure
- `classifier.py` → ML model training and evaluation
- `feature_extractor.py` → behavioral feature extraction
- `logger.py` → keyboard and mouse logging
- `realtime_predictor.py` → live cognitive load prediction
- `task_runner.py` → session task management
- `webcam_display.py` → webcam + MediaPipe visualization

## Applications
- Adaptive educational systems
- UX research
- Human-AI interaction
- Productivity monitoring
- Cognitive workload analysis

## License
Academic project for educational and research purposes.
