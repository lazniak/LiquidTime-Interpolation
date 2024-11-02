# LiquidTime - ComfyUI Custom Node

<div align="center">
  <h3>Advanced Time Control for Animation Interpolation</h3>
  <img src="/api/placeholder/800/200" alt="LiquidTime Banner">
</div>

## Overview

LiquidTime is a powerful frame interpolation node for ComfyUI that gives you precise control over animation timing. Like liquid bending to your will, it allows you to shape the flow of time in your animations, creating dynamic speed changes and smooth transitions.

## Features

- 🎯 Smart frame interpolation for smooth motion
- 🎛️ Advanced time control with three curve points
- 🎚️ Adjustable middle point for precise timing shifts
- 🖥️ High-quality frame generation using FILM model
- ⚡ Supports both GPU (fp16) and CPU (fp32) processing

## Installation

1. Navigate to your ComfyUI custom_nodes directory
2. Clone this repository:
```bash
git clone https://github.com/lazniak/LiquidTime-Interpolation.git
```
3. Install requirements:
```bash
cd LiquidTime
pip install -r requirements.txt
```
4. Restart ComfyUI

The model will be downloaded automatically on first use.

## Parameters

| Parameter | Range | Default | Description |
|-----------|--------|---------|-------------|
| target_frames | 2-1000 | 60 | Desired number of output frames |
| start_curve | -100 to 100 | 0 | Control acceleration at the beginning |
| middle_curve | -100 to 100 | 0 | Adjust speed around middle point |
| middle_moment | 0-100 | 50 | Position of middle influence (%) |
| end_curve | -100 to 100 | 0 | Shape timing at the end |

## Common Use Cases

### Smooth Slow Motion
- start_curve: 0
- middle_curve: -50
- middle_moment: 50
- end_curve: 0

### Dynamic Acceleration
- start_curve: -50
- middle_curve: 0
- middle_moment: 50
- end_curve: 50

### Time Shift
- start_curve: 0
- middle_curve: 70
- middle_moment: 30 (or your preferred position)
- end_curve: 0

## Requirements
- ComfyUI
- torch >= 2.0.0
- numpy >= 1.21.0
- pillow >= 9.0.0
- tqdm >= 4.65.0

## Credits

- Frame interpolation model: [FILM: Frame Interpolation for Large Motion](https://github.com/google-research/frame-interpolation)
- PyTorch implementation: [frame-interpolation-pytorch](https://github.com/dajes/frame-interpolation-pytorch)
- Node development: PabloGFX

## Support

If you find this node useful, consider supporting the development:
- YouTube: [https://youtube.com/@lazniak](https://youtube.com/@lazniak)
- Buy Me a Coffee: [https://buymeacoffee.com/eyb8tkx3to](https://buymeacoffee.com/eyb8tkx3to)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
