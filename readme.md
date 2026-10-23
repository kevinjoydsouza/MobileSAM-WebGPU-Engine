# Real-Time In-Browser Segmentation & Multi-Model Detection

An end-to-end, privacy-focused WebAI engine running real-time object segmentation and multi-model detection locally in the browser using **MobileSAM**, **YOLOv8 Nano**, and **MobileNetV2** via **ONNX Runtime Web** and **WebGPU**.


## Highlights

- **Decoupled Edge AI Pipeline**: Runs a heavy Vision Transformer encoder on static/frozen frames and streams an ultra-fast Mask Decoder at sub-10ms latency using mouse cursor coordinates.
- **WebGPU + WASM Hybrid Engine**: Automatically allocates workloads across WebGPU and SIMD WebAssembly based on tensor buffer constraints.
- **Live Detection Switching**: Seamlessly toggle between **YOLOv8 Nano** (COCO-80 detection matching) and **MobileNetV2** (ImageNet-1K classification) directly from the UI.
- **Configurable Visual Layers**: Independently toggle segmentation masks, corner-bracket bounding boxes, detection labels, and point prompt reticles.

---

## Directory Structure

```text
├── index.html            # Web interface, Tailwind layout, and canvas layers
├── app.js                # Core WebAI orchestrator (ONNX Runtime Web + WebGPU)
├── export_script.py      # Unified export & quantization script
├── requirements.txt      # Python export dependencies
├── coco_classes.js       # Auto-generated COCO-80 label dictionary
├── imagenet_classes.js   # Auto-generated ImageNet-1K label dictionary
└── models/               # Target directory for exported ONNX graphs
    ├── mobile_sam_encoder.onnx
    ├── mobile_sam_decoder.onnx
    ├── yolov8n.onnx
    └── mobilenet.onnx
## Quickstart Guide
1. Environment Setup
Clone the repository and install the export dependencies:

Bash
git clone [https://github.com/your-username/webgpu-mobilesam-detection.git](https://github.com/your-username/webgpu-mobilesam-detection.git)]
git clone 
cd webgpu-mobilesam-detection

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
2. Export & Quantize ONNX Models
Run the unified export script to download weights, trace PyTorch architectures, and export models with correct dynamic axes:

Bash
python export_script.py
3. Serve the Application
WebGPU and camera APIs require a secure origin (localhost or https). Start a local HTTP server:

Bash
# Using Python:
python -m http.server 8000

# Or using Node.js:
npx serve .
Open http://localhost:8000 in Google Chrome (v113+) or Microsoft Edge with WebGPU enabled.

Tech Stack
Model Execution: ONNX Runtime Web (WebGPU / WASM Execution Providers)

Vision Foundation Models: MobileSAM, Ultralytics YOLOv8, MobileNetV2

Client Runtime: Pure Vanilla JavaScript (ES6+), HTML5 Canvas 2D API, Web Media Capture API

Styling: Tailwind CSS

