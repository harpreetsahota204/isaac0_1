# Isaac-0.1 FiftyOne Model Zoo Integration

A [FiftyOne Model Zoo](https://docs.voxel51.com/user_guide/model_zoo/index.html) integration for [Isaac-0.1](https://huggingface.co/PerceptronAI/Isaac-0.1) by Perceptron AI - an open-source, 2B-parameter perceptive-language model designed for real-world visual understanding tasks.

## Overview

Isaac-0.1 is the first in Perceptron AI's family of models built to be the intelligence layer for the physical world. This integration allows you to seamlessly use Isaac-0.1 within FiftyOne for various computer vision tasks including object detection, classification, OCR, visual question answering, and more.

## Features

- **🎯 Object Detection**: Detect and localize objects with bounding boxes

- **📍 Keypoint Detection**: Identify key points in images with spatial awareness

- **📝 OCR**: Extract and detect text from images

- **🏷️ Classification**: Classify images into categories

- **❓ Visual Question Answering**: Answer questions about image content

- **🚀 Efficient**: 2B parameters delivering capabilities comparable to models 50x larger

## Installation

### Prerequisites

```bash
pip install fiftyone
pip install perceptron
pip install transformers
pip install torch torchvision
pip install huggingface-hub
```

### Register the Model Zoo

```python
import fiftyone.zoo as foz

# Register this model zoo source
foz.register_zoo_model_source(
    "https://github.com/harpreetsahota204/isaac0_1",
    overwrite=True
)
```

## Usage

### Loading the Model

```python
import fiftyone.zoo as foz

# Load the Isaac-0.1 model
model = foz.load_zoo_model("PerceptronAI/Isaac-0.1")
```

### Object Detection

```python
import fiftyone as fo
import fiftyone.zoo as foz

# Load a dataset
dataset = foz.load_zoo_dataset("quickstart", max_samples=10)

# Load model and set operation
model = foz.load_zoo_model("PerceptronAI/Isaac-0.1")
model.operation = "detect"

# Set detection prompt
model.prompt = "Animals, Humans, Vehicles, Objects"

# Apply model to dataset
dataset.apply_model(model, label_field="detections")
```

### Visual Question Answering (VQA)

```python
model.operation = "vqa"
model.prompt = "Describe the spatial relationships between objects in this scene"

dataset.apply_model(model, label_field="vqa_response")
```

### OCR Text Detection

```python
model.operation = "ocr_detection"
model.prompt = "Detect all text in this image"

dataset.apply_model(model, label_field="text_detections")
```

### OCR Text Extraction

```python
model.operation = "ocr"
model.prompt = "Extract all text from this image"

dataset.apply_model(model, label_field="extracted_text")
```

### Keypoint Detection

```python
model.operation = "point"
model.prompt = "Identify key features: eyes, nose, corners"

dataset.apply_model(model, label_field="keypoints")
```

### Classification

```python
model.operation = "classify"
model.prompt = "Classify the weather: sunny, rainy, snowy, cloudy, indoor"

dataset.apply_model(model, label_field="weather")
```

## Advanced Usage

### Using Sample-Level Prompts

You can use different prompts for each sample in your dataset:

```python

# Apply model using the prompt field
model.operation = "detect"
dataset.apply_model(
    model, 
    label_field="custom_detections",
    prompt_field="sample_prompt"
)
```

### Custom System Prompts

You can customize the system prompt for specific use cases:

```python
model.system_prompt = """
You are a specialized assistant for medical image analysis.
Focus on identifying anatomical structures and abnormalities.
"""
```

## Model Operations

| Operation | Description | Output Type | Example Prompt |
|-----------|-------------|-------------|----------------|
| `detect` | Object detection with bounding boxes | `fo.Detections` | "Cars, pedestrians, traffic signs" |
| `point` | Keypoint detection | `fo.Keypoints` | "Eyes, nose, mouth corners" |
| `classify` | Image classification | `fo.Classifications` | "Indoor or outdoor scene" |
| `ocr` | Text extraction | String | "Extract all text from the image" |
| `ocr_detection` | Text detection with boxes | `fo.Detections` | "Detect text regions" |
| `vqa` | Visual question answering | String | "What is the main subject?" |

## Example Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/harpreetsahota204/isaac0_1/blob/main/isaac_0_1_example.ipynb)

See [`isaac_0_1_example.ipynb`](isaac_0_1_example.ipynb) for a complete interactive notebook demonstrating all operations. You can run it directly in Google Colab or download it for local execution.

For the original testing script, see [`testing_isaac_0_1.py`](testing_isaac_0_1.py).

## Model Details

- **Parameters**: 2.57B
- **Architecture**: Based on Qwen3-1.7B with custom vision encoder
- **Vision Resolution**: Dynamic, up to 60 megapixels
- **Context Length**: 16,384 tokens
- **Training**: Perceptive-language pretraining on multimodal data

## Performance

Isaac-0.1 delivers strong performance across various benchmarks:

- Visual question answering with reproducible training

- Grounded spatial intelligence with precise localization

- In-context learning without fine-tuning

- Robust OCR across resolutions

- Conversational pointing for auditable reasoning

## License

- **Code**: Apache 2.0 License
- **Model Weights**: Creative Commons Attribution-NonCommercial 4.0 International License

## Citation

If you use Isaac-0.1 in your research or applications, please cite:

```bibtex
@software{isaac2025fiftyone,
  title = {Isaac-0.1 FiftyOne Model Zoo Integration},
  author = {Harpreet Sahota and Perceptron AI},
  year = {2025},
  url = {https://github.com/harpreetsahota204/isaac0_1},
  note = {FiftyOne integration for Isaac-0.1 perceptive-language model}
}

@misc{perceptronai2025isaac,
  title = {Isaac-0.1: A 2B Parameter Perceptive-Language Model},
  author = {{Perceptron AI}},
  year = {2025},
  publisher = {Hugging Face},
  url = {https://huggingface.co/PerceptronAI/Isaac-0.1},
  note = {Open-source multimodal model for real-world visual understanding}
}
```

## Acknowledgments

- [Perceptron AI](https://perceptron.ai) for developing Isaac-0.1

- [FiftyOne](https://voxel51.com/fiftyone) by Voxel51 for the model zoo framework
