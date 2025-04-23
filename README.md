![DCV.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/b3meMjfW6qOwWkuE-UCKQ.png)

# **Multilabel-GeoSceneNet**

> **Multilabel-GeoSceneNet** is a vision-language encoder model fine-tuned from **google/siglip2-base-patch16-224** for **multi-label** image classification. It is designed to recognize and label multiple geographic or environmental elements in a single image using the **SiglipForImageClassification** architecture.

```py
Classification Report:
                          precision    recall  f1-score   support

Buildings and Structures     0.8881    0.9498    0.9179      2190
                  Desert     0.9649    0.9480    0.9564      2000
             Forest Area     0.9807    0.9855    0.9831      2271
        Hill or Mountain     0.8616    0.8993    0.8800      2512
             Ice Glacier     0.9114    0.8382    0.8732      2404
            Sea or Ocean     0.9328    0.9525    0.9426      2274
             Street View     0.9476    0.9106    0.9287      2382

                accuracy                         0.9245     16033
               macro avg     0.9267    0.9263    0.9260     16033
            weighted avg     0.9253    0.9245    0.9244     16033
```

![download.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/Ld-vFb2MWg43wAG5pyFZb.png)

---

The model predicts the presence of one or more of the following **7 geographic scene categories**:

```
    Class 0: "Buildings and Structures"
    Class 1: "Desert"
    Class 2: "Forest Area"
    Class 3: "Hill or Mountain"
    Class 4: "Ice Glacier"
    Class 5: "Sea or Ocean"
    Class 6: "Street View"
```

---

## **Install dependencies**

```python
!pip install -q transformers torch pillow gradio
```

---

## **Inference Code**

```python
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
```

---

## **Intended Use:**

The **Multilabel-GeoSceneNet** model is suitable for recognizing multiple geographic and structural elements in a single image. Use cases include:

- **Remote Sensing:** Label elements in satellite or drone imagery.
- **Geographic Tagging:** Auto-tagging images for search or sorting.
- **Environmental Monitoring:** Identify features like glaciers or forests.
- **Scene Understanding:** Help autonomous systems interpret complex scenes.
