# Android Sensor Data Collection Application

**A comprehensive Android app for collecting touch events and sensor data using Shizuku privileges**
<img width="540" height="1170" alt="app_run_chart" src="https://github.com/user-attachments/assets/3e63a4db-4ff3-4516-842f-7afee0b9c8a8" />

## Features

- **Touch Event Monitoring**: Real-time capture of touch screen events using Linux getevent command
- **Multi-Sensor Data Collection**: Accelerometer, Gyroscope, and Magnetic Field sensors
- **Pattern Lock Simulation**: Custom pattern lock view for testing touch interactions
- **Data Export**: Export collected data to CSV format with timestamps
- **Real-time Display**: Live visualization of sensor data and touch coordinates
- **Configurable Sampling Rate**: Adjustable data collection frequency
- **Current Activity Detection**: Automatically detects the current focused app

## Technical Overview

### Architecture
- **Language**: Java
- **Min SDK**: 26 (Android 8.0)
- **Target SDK**: 35 (Android 15)
- **Privileges**: Requires Shizuku for system-level access
- **Communication**: Socket-based IPC between UI and privileged service

### Data Collection

The app collects comprehensive sensor and touch data including:

**Touch Events:**
- X/Y coordinates
- Pressure values
- Touch action (down/move/up)
- Major/Minor axis values

**Sensor Data:**
- Accelerometer (X, Y, Z axis)
- Gyroscope (X, Y, Z axis)
- Magnetic field (X, Y, Z axis)

**System Information:**
- Timestamps
- Current focused application
- Touch event types

## Requirements

### System Requirements
- Android 8.0+ (API level 26)
- Root access or Shizuku framework installed
- Internet permission for socket communication

### Dependencies
- **Shizuku API**: `dev.rikka.shizuku:api:13.1.5`
- **Shizuku Provider**: `dev.rikka.shizuku:provider:13.1.5`
- **Material Components**: `com.google.android.material:material:1.12.0`
- **NiftySlider**: `io.github.litao0621:nifty-slider-effect:1.4.6`
- **AndroidX Activity**: `androidx.activity:activity:1.9.3`

## Setup Instructions

### 1. Install Shizuku
1. Download and install [Shizuku](https://shizuku.rikka.app/) from GitHub or Google Play
2. Enable Shizuku service using one of these methods:
   - **ADB Method** (Recommended): Connect via USB debugging and run ADB commands
   - **Root Method**: Grant root access to Shizuku
   - **Wireless ADB**: Enable wireless debugging on Android 11+

### 2. Configure App Permissions
1. Install the Sensor Data Collector app
2. Launch the app and grant Shizuku permission when prompted
3. Ensure the following permissions are granted:
   - Internet access
   - Network state access
   - High sampling rate sensors
   - Wake lock
   - External storage write access

### 3. Usage
1. **Start Recording**: Tap "START RECORD" to begin data collection
2. **Monitor Data**: View real-time sensor data and touch events on screen
3. **Interact**: Touch the screen or use the pattern lock to generate events
4. **Stop Recording**: Tap "STOP RECORD" to end data collection
5. **Export Data**: Tap "EXPORT TO CSV" to save collected data

## Data Format

The exported CSV contains the following columns:
```
Date, X, Y, Pressure, Action, MajorAxis, MinorAxis, AccelX, AccelY, AccelZ, GravityX, GravityY, GravityZ, MagneticX, MagneticY, MagneticZ, Activity
```

Sample data row:
```
2025-08-27 14:30:25:123, 540.5, 960.2, 1.0, DOWN, 45.2, 35.1, 0.15, 9.81, 0.03, 0.12, 9.82, 0.05, 23.4, -12.1, 45.6, com.android.launcher
```

## Technical Implementation Details

### Socket Communication
- **Protocol**: TCP
- **Port**: 11451 (localhost)
- **Data Flow**: GetEventService → Socket → MainActivity

### Thread Management
- UI thread handles user interactions
- Background service threads for sensor data collection
- Separate thread for Shizuku service communication

### Memory Management
- Uses Vector for thread-safe data storage
- Implements ReentrantLock for synchronized file operations
- WakeLock management to prevent battery optimization

## Troubleshooting

### Common Issues

**"Need Shizuku permission" error:**
- Ensure Shizuku is installed and running
- Grant permission when prompted
- Check Shizuku service status in Shizuku app

**No touch events captured:**
- Verify Shizuku has proper permissions
- Check if device supports getevent command
- Ensure socket communication is not blocked

**Sensor data not collecting:**
- Check sensor availability on device
- Verify sensor permissions are granted
- Ensure background processing is not restricted

### Debugging
- Enable USB debugging for ADB logs
- Check logcat for "kotorinminami" and "DeviceSensorService" tags
- Monitor Shizuku service status
---
# Authentication Model
A PyTorch implementation of the A2AUTH authentication model for continuous, application-agnostic smartphone user verification.
It fuses multimodal behavioral features (motion sensors, device attitude, touch/gesture features) and learns a discriminative embedding via a residual MLP with triplet loss and semi-hard negative mining.
