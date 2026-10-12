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
