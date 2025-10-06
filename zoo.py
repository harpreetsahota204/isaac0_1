
import os
import logging
import json
import re
from PIL import Image
from typing import Dict, Any, List, Union, Optional

import numpy as np
import torch

import fiftyone as fo
from fiftyone import Model, SamplesMixin

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from transformers.utils.import_utils import is_flash_attn_2_available

from .modular_isaac import IsaacProcessor

logger = logging.getLogger(__name__)

DEFAULT_DETECTION_SYSTEM_PROMPT = """You are a helpful assistant specializing visual grounding for object detection and counting.

Return each detection using this format:

<point_box mention="object label"> (x1,y1) (x2,y2) </point_box>

Or for multiple instances of the same type:

<collection mention="object label">
  <point_box> (x1,y1) (x2,y2) </point_box>
  <point_box> (x1,y1) (x2,y2) </point_box>
</collection>

Detect all relevant objects and provide their labels based on the user's request.

<hint>BOX</hint>
"""

DEFAULT_CLASSIFICATION_SYSTEM_PROMPT = """You are a helpful assistant. You specializes in comprehensive classification across any visual domain, capable of analyzing:

Unless specifically requested for single-class output, multiple relevant classifications can be provided.

Always return your response as valid JSON wrapped in ```json blocks.

```json
{
    "classifications": [
        {
            "label": "descriptive class label",
        }
    ]
}
```

The JSON should contain a list of classifications where:
- Each classification must have a 'label' field
- Labels should be descriptive strings describing what you've identified in the image, but limited to one or two word responses
- The response should be a list of classifications
"""


DEFAULT_KEYPOINT_SYSTEM_PROMPT = """You are a helpful assistant specializing visual grounding for pointing at and counting objects.

Return each keypoint using this format:

<point mention="point label"> (x,y) </point>

Or for multiple instances of the same type:

<collection mention="point label">
  <point> (x1,y1) </point>
  <point> (x2,y2) </point>
  <point> (x3,y3) </point>
</collection>

Point to all relevant objects and provide their labels based on the user's request.

<hint>POINT</hint>
"""

DEFAULT_POLYGON_SYSTEM_PROMPT = """You are a helpful assistant specializing visual grounding for drawing polygons around objects.

Return each polygon using this format:

<polygon mention="polygon label"> (x1,y1) (x2,y2) (x3,y3) (x4,y4) ... </polygon>

Or for multiple instances of the same type:

<collection mention="polygon label">
  <polygon> (x1,y1) (x2,y2) (x3,y3) ... </polygon>
  <polygon> (x1,y1) (x2,y2) (x3,y3) ... </polygon>
</collection>

Draw polygons around all relevant objects and provide their labels based on the user's request.

<hint>POLYGON</hint>
"""

DEFAULT_OCR_DETECTION_SYSTEM_PROMPT = """You are an OCR (Optical Character Recognition) system. Accurately detect, extract, and transcribe all readable text from the image.

Return each text detection using this format, where "text content" is the actual text you detect:

<point_box mention="text content"> (x1,y1) (x2,y2) </point_box>

Detect and read the text in the image.

<hint>BOX</hint>
"""

DEFAULT_OCR_POLYGON_SYSTEM_PROMPT = """You are an OCR (Optical Character Recognition) system. Accurately detect, extract, and transcribe all readable text from the image.

Return each text detection using this format, where "text content" is the actual text you detect:

<polygon mention="text content"> (x1,y1) (x2,y2) (x3,y3) (x4,y4) ... </polygon>

Detect and read the text in the image.

<hint>POLYGON</hint>
"""

DEFAULT_OCR_SYSTEM_PROMPT = """You are an OCR (Optical Character Recognition) system. Accurately detect, extract, and transcribe all readable text from the image. Preserve the original formatting as closely as possible, including:

- Line breaks and paragraphs  
- Headings and subheadings  
- Any tables, lists, bullet points, or numbered items  
- Special characters, spacing, and alignment  

Respond with 'No Text' if there is no text in the provided image.
"""

DEFAULT_VQA_SYSTEM_PROMPT = "You are a visual question answering assistant. Provide a direct, concise answer."

OPERATIONS = {
    "detect": DEFAULT_DETECTION_SYSTEM_PROMPT,
    "point": DEFAULT_KEYPOINT_SYSTEM_PROMPT,
    "ocr_detection": DEFAULT_OCR_DETECTION_SYSTEM_PROMPT,
    "ocr_polygon": DEFAULT_OCR_POLYGON_SYSTEM_PROMPT,
    "classify": DEFAULT_CLASSIFICATION_SYSTEM_PROMPT,
    "segment":DEFAULT_POLYGON_SYSTEM_PROMPT,
    "ocr": DEFAULT_OCR_SYSTEM_PROMPT,
    "vqa": DEFAULT_VQA_SYSTEM_PROMPT
}

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class IsaacModel(SamplesMixin, Model):
    """A FiftyOne model for running Isaac 0.1 vision tasks"""

    def __init__(
        self,
        model_path: str,
        operation: str = None,
        prompt: str = None,
        system_prompt: str = None,
        **kwargs
    ):
        self._fields = {}
        
        self.model_path = model_path
        self._custom_system_prompt = system_prompt  # Store custom system prompt if provided
        self._operation = operation
        self.prompt = prompt
        
        self.device = get_device()
        logger.info(f"Using device: {self.device}")

        model_kwargs = {
            "device_map": self.device,
        }

        # Set optimizations based on CUDA device capabilities
        if self.device == "cuda" and torch.cuda.is_available():
            capability = torch.cuda.get_device_capability(self.device)
            
            # Enable flash attention if available, otherwise use sdpa
            model_kwargs["attn_implementation"] = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
            
            # Enable bfloat16 on Ampere+ GPUs (compute capability 8.0+), otherwise use float16
            model_kwargs["torch_dtype"] = torch.bfloat16 if capability[0] >= 8 else torch.float16

        # Load model and processor
        logger.info(f"Loading model from {model_path}")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            **model_kwargs
        )
        
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            use_fast=False
            )
        
        self.config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True
            )

        logger.info("Loading processor")

        self.processor = IsaacProcessor(
            tokenizer=self.tokenizer,
            config=self.config
        )

    @property
    def needs_fields(self):
        """A dict mapping model-specific keys to sample field names."""
        return self._fields

    @needs_fields.setter
    def needs_fields(self, fields):
        self._fields = fields
    
    def _get_field(self):
        if "prompt_field" in self.needs_fields:
            prompt_field = self.needs_fields["prompt_field"]
        else:
            prompt_field = next(iter(self.needs_fields.values()), None)

        return prompt_field

    @property
    def media_type(self):
        return "image"
    
    @property
    def ragged_batches(self):
        """Enable handling of varying image sizes in batches."""
        return True
    

    @property
    def operation(self):
        return self._operation

    @operation.setter
    def operation(self, value):
        if value not in OPERATIONS:
            raise ValueError(f"Invalid operation: {value}. Must be one of {list(OPERATIONS.keys())}")
        self._operation = value

    @property
    def system_prompt(self):
        # Return custom system prompt if set, otherwise return default for current operation
        return self._custom_system_prompt if self._custom_system_prompt is not None else OPERATIONS[self.operation]

    @system_prompt.setter
    def system_prompt(self, value):
        self._custom_system_prompt = value


    def _strip_think_blocks(self, text: str) -> str:
        """Remove <think>...</think> blocks from text."""
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    def _parse_coordinates(self, coord_str: str) -> Optional[List[float]]:
        """Extract coordinates from various formats like (x,y) or (x1,y1) (x2,y2)."""
        # Find all numbers (int or float) in the string
        numbers = re.findall(r'-?\d+(?:\.\d+)?', coord_str)
        if numbers:
            return [float(n) for n in numbers]
        return None

    def _extract_point_boxes(self, text: str) -> List[Dict]:
        """Extract all <point_box> elements from text."""
        boxes = []
        
        # Pattern for point_box with optional mention attribute
        pattern = r'<point_box(?:\s+mention="([^"]*)")?\s*>\s*(.*?)\s*</point_box>'
        
        for match in re.finditer(pattern, text, re.DOTALL):
            mention = match.group(1)  # May be None
            coords_text = match.group(2)
            
            coords = self._parse_coordinates(coords_text)
            if coords and len(coords) >= 4:
                boxes.append({
                    'bbox_2d': coords[:4],  # Take first 4 numbers
                    'label': mention or 'object'
                })
        
        return boxes

    def _extract_points(self, text: str) -> List[Dict]:
        """Extract all <point> elements from text."""
        points = []
        
        # Pattern for point with optional mention attribute
        pattern = r'<point(?:\s+mention="([^"]*)")?\s*>\s*(.*?)\s*</point>'
        
        for match in re.finditer(pattern, text, re.DOTALL):
            mention = match.group(1)  # May be None
            coords_text = match.group(2)
            
            coords = self._parse_coordinates(coords_text)
            if coords and len(coords) >= 2:
                points.append({
                    'point_2d': coords[:2],  # Take first 2 numbers
                    'label': mention or 'point'
                })
        
        return points

    def _extract_polygons(self, text: str) -> List[Dict]:
        """Extract all <polygon> elements from text."""
        polygons = []
        
        # Pattern for polygon with optional mention attribute
        pattern = r'<polygon(?:\s+mention="([^"]*)")?\s*>\s*(.*?)\s*</polygon>'
        
        for match in re.finditer(pattern, text, re.DOTALL):
            mention = match.group(1)  # May be None
            coords_text = match.group(2)
            
            coords = self._parse_coordinates(coords_text)
            if coords and len(coords) >= 6:  # At least 3 points (6 numbers)
                # Group coordinates into pairs
                vertices = []
                for i in range(0, len(coords) - 1, 2):
                    vertices.append([coords[i], coords[i + 1]])
                
                polygons.append({
                    'vertices': vertices,
                    'label': mention or 'polygon'
                })
        
        return polygons

    def _extract_all_elements(self, text: str, element_type: str = None) -> List[Dict]:
        """Extract all elements from text, both from collections and standalone, with proper inheritance."""
        all_elements = []
        
        # Pattern for collection with optional mention attribute
        collection_pattern = r'<collection(?:\s+mention="([^"]*)")?\s*>(.*?)</collection>'
        
        # First, process collections
        for match in re.finditer(collection_pattern, text, re.DOTALL):
            collection_mention = match.group(1)
            collection_content = match.group(2)
            
            # Extract elements from within the collection
            if element_type == 'point_box' or element_type is None:
                boxes = self._extract_point_boxes(collection_content)
                for box in boxes:
                    # Inherit collection mention if box doesn't have one
                    if box['label'] == 'object' and collection_mention:
                        box['label'] = collection_mention
                    all_elements.append({'bbox_2d': box['bbox_2d'], 'label': box['label']})
            
            if element_type == 'point' or element_type is None:
                points = self._extract_points(collection_content)
                for point in points:
                    # Inherit collection mention if point doesn't have one
                    if point['label'] == 'point' and collection_mention:
                        point['label'] = collection_mention
                    all_elements.append({'point_2d': point['point_2d'], 'label': point['label']})
            
            if element_type == 'polygon' or element_type is None:
                polygons = self._extract_polygons(collection_content)
                for polygon in polygons:
                    # Inherit collection mention if polygon doesn't have one
                    if polygon['label'] == 'polygon' and collection_mention:
                        polygon['label'] = collection_mention
                    all_elements.append({'vertices': polygon['vertices'], 'label': polygon['label']})
        
        # Remove collections from text to avoid double-processing
        text_without_collections = re.sub(collection_pattern, '', text, flags=re.DOTALL)
        
        # Process standalone elements (not in collections)
        if element_type == 'point_box' or element_type is None:
            all_elements.extend(self._extract_point_boxes(text_without_collections))
        if element_type == 'point' or element_type is None:
            all_elements.extend(self._extract_points(text_without_collections))
        if element_type == 'polygon' or element_type is None:
            all_elements.extend(self._extract_polygons(text_without_collections))
        
        return all_elements

    def _parse_model_output(self, output_text: str) -> Dict:
        """
        Unified parser that handles both XML-like and JSON formats.
        Returns a standardized dictionary structure regardless of input format.
        """
        # Step 1: Clean the output
        cleaned_text = self._strip_think_blocks(output_text)
        
        # Step 2: Detect format and parse accordingly
        # Check if it's XML-like format
        has_xml_tags = any(tag in cleaned_text for tag in ['<point>', '<point_box>', '<polygon>', '<collection>'])
        
        if has_xml_tags:
            # Parse XML-like format
            result = {
                'detections': [],
                'keypoints': [],
                'polygons': [],
                'text_detections': [],
                'text_polygons': [],
                'classifications': []
            }
            
            # Extract all elements (both from collections and standalone)
            all_elements = self._extract_all_elements(cleaned_text)
            
            # Sort elements into appropriate categories
            for elem in all_elements:
                if 'bbox_2d' in elem:
                    result['detections'].append(elem)
                    # Also add to text_detections for OCR operations
                    # Using 'label' consistently instead of 'text' field
                    result['text_detections'].append(elem)
                elif 'point_2d' in elem:
                    result['keypoints'].append(elem)
                elif 'vertices' in elem:
                    result['polygons'].append(elem)
                    # Also add to text_polygons for OCR polygon operations
                    # Using 'label' consistently for the detected text
                    result['text_polygons'].append(elem)
            
            return result
        else:
            # Try JSON format parsing
            json_text = cleaned_text
            
            # Handle JSON wrapped in markdown code blocks
            if "```json" in json_text:
                try:
                    # Extract JSON between ```json and ``` markers
                    json_text = json_text.split("```json")[1].split("```")[0].strip()
                except:
                    pass
            
            # Attempt to parse the JSON string
            try:
                parsed_json = json.loads(json_text)
                # Ensure all expected keys exist
                if isinstance(parsed_json, dict):
                    for key in ['detections', 'keypoints', 'polygons', 'classifications', 'text_detections', 'text_polygons']:
                        if key not in parsed_json:
                            parsed_json[key] = []
                    return parsed_json
            except:
                # Log first 200 chars of failed parse for debugging
                logger.debug(f"Failed to parse as JSON: {json_text[:200]}")
            
            # If both fail, return empty structure
            logger.debug(f"Could not parse output in any known format: {cleaned_text[:200]}")
            return {
                'detections': [],
                'keypoints': [],
                'polygons': [],
                'classifications': [],
                'text_detections': [],
                'text_polygons': []
            }

    def _to_detections(self, boxes: List[Dict]) -> fo.Detections:
        """Convert bounding boxes to FiftyOne Detections.
        
        This method handles both regular object detections and OCR text detections.
        
        Args:
            boxes: List of dictionaries containing bounding box info.
                Each box should have:
                - 'bbox_2d' or 'bbox': List of [x1,y1,x2,y2] coordinates in 0-1000 range from model
                - 'label': String label (defaults to "object"). For OCR detections, this contains the detected text.

        Returns:
            fo.Detections object containing the converted bounding box annotations
            with coordinates normalized to [0,1] x [0,1] range
        """
        if not boxes:
            return fo.Detections(detections=[])
            
        detections = []
        
        for box in boxes:
            try:
                # Try to get bbox from either bbox_2d or bbox field
                bbox = box.get('bbox_2d', box.get('bbox', None))
                if not bbox:
                    continue
                    
                # Model outputs coordinates in 0-1000 range, normalize to 0-1
                x1, y1, x2, y2 = map(float, bbox)
                x = x1 / 1000.0  # Normalized left x
                y = y1 / 1000.0  # Normalized top y
                w = (x2 - x1) / 1000.0  # Normalized width
                h = (y2 - y1) / 1000.0  # Normalized height
                
                # Create Detection object with normalized coordinates
                try:
                    detection = fo.Detection(
                        label=str(box.get("label", "object")),
                        bounding_box=[x, y, w, h],
                    )
                    detections.append(detection)
                except:
                    continue
                
            except Exception as e:
                # Log any errors processing individual boxes but continue
                logger.debug(f"Error processing box {box}: {e}")
                continue
                
        return fo.Detections(detections=detections)


    def _to_keypoints(self, points: List[Dict]) -> fo.Keypoints:
        """Convert a list of point dictionaries to FiftyOne Keypoints.
        
        Args:
            points: List of dictionaries containing point information.
                Each point should have:
                - 'point_2d': List of [x,y] coordinates in 0-1000 range from model
                - 'label': String label describing the point
                
        Returns:
            fo.Keypoints object containing the converted keypoint annotations
            with coordinates normalized to [0,1] x [0,1] range
        """
        if not points:
            return fo.Keypoints(keypoints=[])
            
        keypoints = []
        
        for point in points:
            try:
                # Get coordinates from point_2d field and convert to float
                x, y = point["point_2d"]
                x = float(x)
                y = float(y)
                
                # Model outputs coordinates in 0-1000 range, normalize to 0-1
                normalized_point = [
                    x / 1000.0,
                    y / 1000.0
                ]
                
                keypoint = fo.Keypoint(
                    label=str(point.get("label", "point")),
                    points=[normalized_point],
                )
                keypoints.append(keypoint)
            except Exception as e:
                logger.debug(f"Error processing point {point}: {e}")
                continue
                
        return fo.Keypoints(keypoints=keypoints)

    def _to_polygons(self, polygons: List[Dict]) -> fo.Polylines:
        """Convert polygon data to FiftyOne Polylines.
        
        This method handles both regular polygon segmentations and OCR text polygons.
        
        Args:
            polygons: List of dictionaries containing polygon information.
                Each dictionary should have:
                - 'vertices': List of [x,y] coordinate pairs in 0-1000 range
                - 'label': String label describing the polygon. For OCR polygons, this contains the detected text.
                
        Returns:
            fo.Polylines object containing the polygon annotations
        """
        if not polygons:
            return fo.Polylines(polylines=[])
            
        polylines = []
        
        for polygon in polygons:
            try:
                vertices = polygon.get('vertices', [])
                if not vertices or len(vertices) < 3:
                    continue
                    
                # Convert vertices from 0-1000 range to 0-1 normalized
                normalized_points = []
                for x, y in vertices:
                    norm_x = float(x) / 1000.0
                    norm_y = float(y) / 1000.0
                    normalized_points.append([norm_x, norm_y])
                
                # Create a Polyline - points should be a list of shapes, 
                # where each shape is a list of [x,y] points
                polyline = fo.Polyline(
                    label=str(polygon.get('label', 'polygon')),
                    points=[normalized_points],  # Wrap in list since it's one polygon
                    closed=True,  # Polygons are closed shapes
                    filled=True   # Can be used as segmentation masks
                )
                polylines.append(polyline)
                
            except Exception as e:
                logger.debug(f"Error processing polygon {polygon}: {e}")
                continue
                
        return fo.Polylines(polylines=polylines)

    def _to_classifications(self, classes: List[Dict]) -> fo.Classifications:
        """Convert a list of classification dictionaries to FiftyOne Classifications.
        
        Args:
            classes: List of dictionaries containing classification information.
                Each dictionary should have:
                - 'label': String class label
                
        Returns:
            fo.Classifications object containing the converted classification 
            annotations with labels
        """
        if not classes:
            return fo.Classifications(classifications=[])
            
        classifications = []
        
        # Process each classification dictionary
        for cls in classes:
            try:
                # Create Classification object with required label and optional confidence
                classification = fo.Classification(
                    label=str(cls["label"]),  # Convert label to string for consistency
                )
                classifications.append(classification)
            except Exception as e:
                # Log any errors but continue processing remaining classifications
                logger.debug(f"Error processing classification {cls}: {e}")
                continue
                
        # Return Classifications container with all processed results
        return fo.Classifications(classifications=classifications)


    def _process_output(self, output_text: str, image: Image.Image):
        """Process model output text based on the current operation type.
        
        Args:
            output_text: Raw text output from the model
            image: PIL Image that was processed (needed for size information)
            
        Returns:
            Processed output in the appropriate format for the operation:
            - str for vqa and ocr operations
            - fo.Detections for detect and ocr_detection operations
            - fo.Keypoints for point operations
            - fo.Classifications for classify operations
            - None if operation is not recognized
        """
        if self.operation == "vqa":
            # VQA returns plain text after stripping think blocks
            cleaned = self._strip_think_blocks(output_text)
            return cleaned.strip()
        
        elif self.operation == "ocr":
            # OCR returns plain text after stripping think blocks
            cleaned = self._strip_think_blocks(output_text)
            return cleaned.strip()
        
        elif self.operation == "detect":
            parsed = self._parse_model_output(output_text)
            data = parsed.get('detections', [])
            return self._to_detections(data)
        
        elif self.operation == "point":
            parsed = self._parse_model_output(output_text)
            data = parsed.get('keypoints', [])
            
            # Special case: if model outputs point_boxes when asked for points,
            # convert them to keypoints (use center of box)
            if not data and parsed.get('detections'):
                for detection in parsed['detections']:
                    if 'bbox_2d' in detection:
                        x1, y1, x2, y2 = detection['bbox_2d']
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        data.append({
                            'point_2d': [center_x, center_y],
                            'label': detection.get('label', 'point')
                        })
            
            return self._to_keypoints(data)
        
        elif self.operation == "classify":
            # Classifications might still come as JSON
            parsed = self._parse_model_output(output_text)
            data = parsed.get('classifications', [])
            return self._to_classifications(data)
        
        elif self.operation == "ocr_detection":
            parsed = self._parse_model_output(output_text)
            # For OCR detection, the text is stored in the label field
            data = parsed.get('text_detections', [])
            return self._to_detections(data)
        
        elif self.operation == "ocr_polygon":
            parsed = self._parse_model_output(output_text)
            # For OCR polygon detection, the text is stored in the label field
            data = parsed.get('text_polygons', [])
            return self._to_polygons(data)
        
        elif self.operation == "polygon" or self.operation == "segment":
            # Add polygon/segmentation operation support
            parsed = self._parse_model_output(output_text)
            data = parsed.get('polygons', [])
            return self._to_polygons(data)
        
        else:
            return None

    def _predict(self, image: Image.Image, sample=None) -> Union[fo.Detections, fo.Keypoints, fo.Classifications, str]:
        """Process a single image through the model and return predictions.
        
        This internal method handles the core prediction logic including:
        - Constructing the chat messages with system prompt and user query
        - Processing the image and text through the model
        - Parsing the output based on the operation type (detection/points/classification/VQA)
        
        Args:
            image: PIL Image to process
            sample: Optional FiftyOne sample containing the image filepath
            
        Returns:
            One of:
            - fo.Detections: For object detection results
            - fo.Keypoints: For keypoint detection results  
            - fo.Classifications: For classification results
            - str: For VQA text responses
            
        Raises:
            ValueError: If no prompt has been set
        """
        # Use local prompt variable instead of modifying self.prompt
        prompt = self.prompt  # Start with instance default
        
        if sample is not None and self._get_field() is not None:
            field_value = sample.get_field(self._get_field())
            if field_value is not None:
                prompt = str(field_value)  # Local variable, doesn't affect instance

        # Prepare input
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": "<image>"},
            {"role": "user", "content": prompt}
        ]
        images = [image]  # Replace with your image path

        # Process input
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
            )

        inputs = self.processor(
            text=text, 
            images=images, 
            return_tensors="pt"
            )
        
        tensor_stream = inputs["tensor_stream"].to(self.device)

        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                tensor_stream=tensor_stream,
                max_new_tokens=16384,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )

        # Decode and process output
        output_text = self.processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return self._process_output(output_text, image)

    def _predict_batch(self, images: List[Image.Image], samples: Optional[List] = None) -> List:
        """Process multiple images in a single model call for efficiency.
        
        Args:
            images: List of PIL Images to process
            samples: Optional list of FiftyOne samples corresponding to each image
            
        Returns:
            List of predictions, one for each input image
        """
        batch_size = len(images)
        results = []
        
        # Process each image with its corresponding sample (if provided)
        for i in range(batch_size):
            image = images[i]
            sample = samples[i] if samples and i < len(samples) else None
            
            # Get prompt for this specific image/sample
            prompt = self.prompt  # Start with instance default
            
            if sample is not None and self._get_field() is not None:
                field_value = sample.get_field(self._get_field())
                if field_value is not None:
                    prompt = str(field_value)
            
            # Prepare messages for this image
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": "<image>"},
                {"role": "user", "content": prompt}
            ]
            
            # Process input
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=text, 
                images=[image],
                return_tensors="pt"
            )
            
            tensor_stream = inputs["tensor_stream"].to(self.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    tensor_stream=tensor_stream,
                    max_new_tokens=8192,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
            
            # Decode and process output
            output_text = self.processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            result = self._process_output(output_text, image)
            results.append(result)
        
        return results

    def predict_all(self, args):
        """Efficient batch prediction for multiple images.
        
        This method enables batch processing of multiple images for improved
        performance when processing datasets.
        
        Args:
            args: List of tuples where each tuple contains (image, sample) or just images
            
        Returns:
            List of predictions, one for each input
        """
        if not args:
            return []
        
        # Separate images and samples
        images = []
        samples = []
        
        for arg in args:
            if isinstance(arg, tuple):
                image, sample = arg
                samples.append(sample)
            else:
                image = arg
                samples.append(None)
            
            # Convert numpy arrays to PIL Images
            if isinstance(image, np.ndarray):
                images.append(Image.fromarray(image))
            else:
                images.append(image)
        
        # Process batch
        return self._predict_batch(images, samples if any(s is not None for s in samples) else None)

    def predict(self, image, sample=None):
        """Process an image with the model.
        
        A convenience wrapper around _predict that handles numpy array inputs
        by converting them to PIL Images first.
        
        Args:
            image: PIL Image or numpy array to process
            sample: Optional FiftyOne sample containing the image filepath
            
        Returns:
            Model predictions in the appropriate format for the current operation
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self._predict(image, sample)