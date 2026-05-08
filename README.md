# korone-fullbody-tracking
Single-camera VR full body tracking built on top of the original MediaPipe VR fullbody project.

## Status
This is an actively modified fork focused on making camera-based tracking feel closer to real trackers.

Current focus:
- Better leg behavior
- Stronger planted-foot stability
- Clearer bending and torso response
- Cleaner setup for SteamVR and VRChat OSC

More features, polish, and proper release packaging are coming soon.

## Credits
`korone-fullbody-tracking` is made by `Korone`.

This project is based on the original `Mediapipe-VR-Fullbody-Tracking` work by its original author and contributors.

Base project credit:
- Original repository: `ju1ce/Mediapipe-VR-Fullbody-Tracking`
- Original concept/workflow and earlier implementation belong to the base project creators

This fork keeps that credit and builds on top of it.

## What This Fork Changes
- Reworked lower-body stabilization
- Added stronger foot locking
- Improved torso lean carry when bending
- Expanded tracker output for chest and knees
- Ongoing cleanup and rebrand

## Planned
- Better calibration flow
- Better defaults for different camera setups
- Cleaner release builds
- Updated UI wording and setup docs
- More tuning for leg and hip behavior

## Running
From this folder:

1. Install dependencies.
2. Run `mediapipepose.bat`.
3. Choose your camera/backend settings.
4. Calibrate in VR.

Main launch files:
- `install_libraries.bat`
- `mediapipepose.bat`
- `pipetest.bat`

## Notes
- SteamVR support is still the main target path.
- VRChat OSC support is included and still being tuned.
- Camera tracking will never be identical to real hardware trackers, but this fork is aimed at closing the gap as much as possible.

## License
Please review the existing [LICENSE](LICENSE) from the base project before redistribution or commercialization.
