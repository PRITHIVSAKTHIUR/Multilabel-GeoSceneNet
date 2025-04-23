import gradio as gr
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Multilabel-GeoSceneNet"  # Updated model name
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

def classify_geoscene_image(image):
    """Predicts geographic scene labels for an input image."""
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits).squeeze().tolist()  # Sigmoid for multilabel
    
    labels = {
        "0": "Buildings and Structures",
        "1": "Desert",
        "2": "Forest Area",
        "3": "Hill or Mountain",
        "4": "Ice Glacier",
        "5": "Sea or Ocean",
        "6": "Street View"
    }
    
    threshold = 0.5
    predictions = {
        labels[str(i)]: round(probs[i], 3)
        for i in range(len(probs)) if probs[i] >= threshold
    }

    return predictions or {"None Detected": 0.0}

# Create Gradio interface
iface = gr.Interface(
    fn=classify_geoscene_image,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Predicted Scene Categories"),
    title="Multilabel-GeoSceneNet",
    description="Upload an image to detect multiple geographic scene elements (e.g., forest, ocean, buildings)."
)

if __name__ == "__main__":
    iface.launch()
