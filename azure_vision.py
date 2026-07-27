import os
import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.exceptions import HttpResponseError

load_dotenv()

endpoint = os.getenv("AZURE_ENDPOINT")
key = os.getenv("AZURE_KEY")

if endpoint and key:
    client = ImageAnalysisClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key)
    )
else:
    client = None


def analyze_image(image_data):
    """Analyze image using Azure AI Vision API."""
    if not client:
        raise ValueError("Azure Vision client is not configured. Please check your .env file.")
    
    if isinstance(image_data, (bytes, bytearray)):
        image_bytes = image_data
    else:
        with open(image_data, "rb") as f:
            image_bytes = f.read()

    try:
        result = client.analyze(
            image_data=image_bytes,
            visual_features=[
                VisualFeatures.CAPTION,
                VisualFeatures.TAGS
            ]
        )
    except HttpResponseError as e:
        if "Caption" in str(e) or "caption" in str(e).lower():
            result = client.analyze(
                image_data=image_bytes,
                visual_features=[
                    VisualFeatures.TAGS
                ]
            )
        else:
            raise e

    return result


def validate_image_quality(result=None):
    """
    Validates if the image is a genuine skin lesion / clinical specimen.
    Returns (is_valid, warning_message, detected_tags).
    """
    tags = []
    if result is not None and hasattr(result, 'tags') and result.tags is not None:
        try:
            if hasattr(result.tags, 'list') and result.tags.list:
                tags = [tag.name.lower() for tag in result.tags.list if hasattr(tag, 'name')]
        except Exception:
            tags = []
    
    # Strict skin lesion / dermatology terms
    lesion_terms = {'skin', 'mole', 'lesion', 'spot', 'scar', 'flesh', 'dermatology', 'epidermis', 'nevus', 'rash', 'tissue', 'wound', 'ulcer', 'close-up', 'macro'}
    
    # Non-medical portrait / general object terms that indicate a non-lesion photo
    portrait_terms = {'clothing', 'shirt', 'suit', 'tie', 'glasses', 'portrait', 'selfie', 'smile', 'happy', 'boy', 'man', 'woman', 'girl', 'child', 'person', 'human', 'wall', 'indoor', 'furniture', 'building', 'car', 'outdoor'}

    has_lesion_context = any(term in tags for term in lesion_terms)
    is_portrait_or_object = any(term in tags for term in portrait_terms)
    
    warnings = []
    
    # If tagged as portrait/clothing/person without explicit skin lesion context
    if is_portrait_or_object and not (('lesion' in tags or 'mole' in tags or 'dermatology' in tags)):
        warnings.append("⚠️ Non-Medical Photo Warning: The uploaded image appears to be a general portrait or object, not a dermoscopic skin lesion. Screening results may be inaccurate.")
    elif not has_lesion_context and len(tags) > 0:
        warnings.append("⚠️ Non-Clinical Specimen Warning: The image does not clearly match a skin lesion specimen. Please ensure you upload a close-up photo of the lesion.")

    # Blur / Low Quality Detection
    poor_quality_terms = {'blur', 'blurry', 'defocus', 'dark', 'low light', 'shadow'}
    detected_quality_issues = [term for term in poor_quality_terms if term in tags]
    if detected_quality_issues:
        warnings.append(f"⚠️ Image Quality Warning: Potential quality issue detected ({', '.join(detected_quality_issues)}). Ensure the photo is sharp and well-lit.")
        
    is_valid = len(warnings) == 0
    warning_text = "\n\n".join(warnings) if warnings else "✅ Image passed clinical quality validation."
    
    return is_valid, warning_text, tags
