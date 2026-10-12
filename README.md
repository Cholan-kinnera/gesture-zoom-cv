<div align="center">

# Gesture-Zoom-cv

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=20&pause=1000&color=2563EB&center=true&vCenter=true&width=650&lines=Real-Time+Gesture-Driven+Computer+Vision+Engine;Sub-pixel+Decoupled+Zoom+%26+Pan+Controller;Powered+by+MediaPipe+Tasks+and+OpenCV;Engineered+by+Cholan+Kinnera" alt="Gesture-Zoom-cv Typing SVG" />
</p>

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=flat-square&logo=python)](https://www.python.org/)
[![MediaPipe Tasks](https://img.shields.io/badge/MediaPipe_Tasks-v0.10.x-green.svg?style=flat-square&logo=google)](https://developers.google.com/mediapipe)
[![OpenCV](https://img.shields.io/badge/OpenCV-v4.8%2B-red.svg?style=flat-square&logo=opencv)](https://opencv.org/)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20POSIX-lightgrey.svg?style=flat-square&logo=linux)](https://endeavouros.com/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg?style=flat-square)](LICENSE)

**Author:** [Cholan Kinnera](https://github.com/Cholan-kinnera)

</div>

---

## Executive Overview

**Gesture-Zoom-cv** is a high-performance, real-time computer vision engine engineered to provide intuitive, touchless spatial manipulation of digital imagery using dual-hand kinematic gestures.

Built on Google's modern **MediaPipe Tasks API** (`vision.HandLandmarker`) and **OpenCV**, this project circumvents the legacy `mp.solutions` deprecation barrier by leveraging the native graph-execution runtime and the official `float16` hand landmarker model.

### Key Capabilities

- **Decoupled Kinematic Control**: Independent relative zoom scaling and spatial panning without positional drift.
- **Orientation-Invariant Geometry**: Robust finger curl and extension classification based on palm-relative Euclidean metrics rather than naive vertical coordinates.
- **Sub-Pixel Exponential Smoothing**: First-order exponential moving average (EMA) interpolation eliminates landmark micro-jitter.
- **Persistent State Tracking**: Zero-jump transformation retention across tracking dropouts and gesture pauses.
- **Fault-Tolerant Boundary Clipping**: Dynamic sub-matrix slicing guarantees that manipulated assets never cause shape-mismatch exceptions when crossing webcam frame boundaries.
- **Presentation Dashboard**: Real-time Heads-Up Display (HUD) presenting tracking telemetry, gesture state, distance, scale, and FPS metrics.

---

## System Architecture & Pipeline

Gesture-Zoom-cv operates as an optimized pipeline structured into five modular abstraction layers:

```mermaid
flowchart TD
    subgraph INGESTION["1. Acquisition Layer"]
        CAM["Webcam Feed (OpenCV VideoCapture)"] --> BGR["Raw BGR Frame (1280x720)"]
    end

    subgraph DETECTION["2. Inference Layer"]
        BGR --> RGB["BGR to RGB Color Conversion"]
        RGB --> MP["MediaPipe Tasks: HandLandmarker (float16)"]
        MP --> LMS["21 Normalized 3D Landmarks Per Hand"]
    end

    subgraph GEOMETRY["3. Kinematic & Gesture Analysis Layer"]
        LMS --> MCP["Compute Palm Center: (MCP5 + MCP17) / 2"]
        LMS --> CLASSIFIER["Geometric Finger State Classifier"]
        CLASSIFIER --> HYST["Temporal Hysteresis Filter (2-on / 3-off)"]
        HYST --> STATE{"Gesture Active?"}
    end

    subgraph TRANSFORM["4. Transformation & Smoothing Engine"]
        STATE -- "Yes" --> CALC_ZOOM["Relative Distance Ratio: d(t) / d_ref"]
        STATE -- "Yes" --> CALC_PAN["Midpoint Translation: M(t) - M_ref"]
        CALC_ZOOM --> EMA_Z["EMA Filter: Zoom (alpha = 0.30)"]
        CALC_PAN --> EMA_P["EMA Filter: Pan (alpha = 0.32)"]
        STATE -- "No" --> HOLD["Persist Last Transform State"]
    end

    subgraph COMPOSITION["5. Rendering & Composition Layer"]
        EMA_Z --> RESIZE["Optimal Interpolation Resize (INTER_AREA / LINEAR)"]
        HOLD --> RESIZE
        RESIZE --> CLIP["Sub-Matrix Safe Intersection Overlay"]
        EMA_P --> CLIP
        CLIP --> HUD["HUD Telemetry & Skeleton Compositor"]
        HUD --> DISP["High-Refresh Window (cv2.imshow)"]
    end
```

### Pipeline Layer Responsibilities

| Layer | Responsibility | Primary Technologies |
| :--- | :--- | :--- |
| **1. Acquisition** | High-definition camera stream capture (`1280x720` resolution). Non-blocking polling prevents frame buffer queuing. | OpenCV `VideoCapture`, V4L2 |
| **2. Inference** | Hand detection and 21 3D normalized keypoint regression via hardware-accelerated CPU delegate. | MediaPipe Tasks `HandLandmarker` (`float16`), XNNPACK |
| **3. Kinematics** | Pixel-space coordinate projection, palm center extraction, and angle/distance invariant finger state evaluation. | Vector Euclidean geometry |
| **4. Transforms** | Continuous distance ratio computation, decoupled midpoint tracking, and low-pass filtering. | Exponential Moving Average (EMA) |
| **5. Composition** | Sub-matrix intersection clipping, dual-mode interpolation resizing, telemetry rendering, and frame display. | NumPy slicing, OpenCV GUI |

---
