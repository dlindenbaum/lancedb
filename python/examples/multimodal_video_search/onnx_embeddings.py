"""
ONNX-based embedding models for efficient inference

This module provides ONNX implementations of CLIP and DINOv3 models,
eliminating the need for PyTorch and providing faster inference with
smaller memory footprint.
"""

import io
import os
from pathlib import Path
from typing import List, Union, Tuple, Optional
import warnings

import numpy as np
from PIL import Image
import onnxruntime as ort


class ONNXModelDownloader:
    """
    Utility to download pre-converted ONNX models
    """

    MODELS = {
        "clip_vit_b32": {
            "visual": "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/vision_model_quantized.onnx",
            "textual": "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/text_model_quantized.onnx",
            "tokenizer": "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/tokenizer.json"
        },
        "clip_vit_b16": {
            "visual": "https://huggingface.co/Xenova/clip-vit-base-patch16/resolve/main/vision_model_quantized.onnx",
            "textual": "https://huggingface.co/Xenova/clip-vit-base-patch16/resolve/main/text_model_quantized.onnx",
            "tokenizer": "https://huggingface.co/Xenova/clip-vit-base-patch16/resolve/main/tokenizer.json"
        },
        "dinov2_vits14": {
            "model": "https://huggingface.co/onnx-community/dinov2-small/resolve/main/model.onnx"
        },
        "dinov2_vitb14": {
            "model": "https://huggingface.co/onnx-community/dinov2-base/resolve/main/model.onnx"
        }
    }

    @staticmethod
    def download_model(model_name: str, cache_dir: str = "~/.cache/onnx_models") -> dict:
        """
        Download ONNX model files

        Args:
            model_name: Name of the model
            cache_dir: Directory to cache downloaded models

        Returns:
            Dict with paths to downloaded model files
        """
        import urllib.request

        cache_dir = Path(cache_dir).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)

        if model_name not in ONNXModelDownloader.MODELS:
            raise ValueError(f"Unknown model: {model_name}")

        model_info = ONNXModelDownloader.MODELS[model_name]
        model_dir = cache_dir / model_name
        model_dir.mkdir(exist_ok=True)

        paths = {}

        for file_type, url in model_info.items():
            filename = url.split("/")[-1]
            filepath = model_dir / filename

            if not filepath.exists():
                print(f"Downloading {model_name} {file_type}...")
                urllib.request.urlretrieve(url, filepath)
                print(f"✓ Downloaded to {filepath}")

            paths[file_type] = str(filepath)

        return paths


class ONNXCLIPEmbeddings:
    """
    ONNX-based CLIP embeddings for efficient inference

    Uses pre-converted ONNX models from HuggingFace for both
    visual and textual encoding without PyTorch dependency.
    """

    def __init__(self, model_name: str = "clip_vit_b32",
                 providers: List[str] = None,
                 cache_dir: str = "~/.cache/onnx_models"):
        """
        Initialize ONNX CLIP model

        Args:
            model_name: Model variant (clip_vit_b32, clip_vit_b16)
            providers: ONNX Runtime providers (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider'])
            cache_dir: Directory to cache models
        """
        if providers is None:
            # Auto-detect best provider
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available_providers:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                print("✓ Using GPU acceleration (CUDA)")
            else:
                providers = ['CPUExecutionProvider']
                print("✓ Using CPU")

        self.providers = providers
        self.model_name = model_name

        # Download models if needed
        print(f"Loading {model_name} ONNX models...")
        model_paths = ONNXModelDownloader.download_model(model_name, cache_dir)

        # Load ONNX models
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.visual_session = ort.InferenceSession(
            model_paths["visual"],
            sess_options=sess_options,
            providers=providers
        )

        self.text_session = ort.InferenceSession(
            model_paths["textual"],
            sess_options=sess_options,
            providers=providers
        )

        # Load tokenizer
        from transformers import CLIPTokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        print(f"✓ Loaded {model_name} ONNX models")

    def preprocess_image(self, image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Preprocess image for CLIP

        Args:
            image: Input image

        Returns:
            Preprocessed image tensor
        """
        # Convert to PIL Image
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image).convert('RGB')
        else:
            img = image.convert('RGB')

        # Resize and normalize (CLIP preprocessing)
        img = img.resize((224, 224), Image.BICUBIC)
        img_array = np.array(img).astype(np.float32) / 255.0

        # Normalize with CLIP stats
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])
        img_array = (img_array - mean) / std

        # Change to CHW format and add batch dimension
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)

        return img_array.astype(np.float32)

    def encode_image(self, image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Encode image to embedding

        Args:
            image: Input image

        Returns:
            Normalized image embedding
        """
        img_tensor = self.preprocess_image(image)

        # Run inference
        inputs = {self.visual_session.get_inputs()[0].name: img_tensor}
        outputs = self.visual_session.run(None, inputs)

        # Normalize
        embedding = outputs[0][0]
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        return embedding.astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text to embedding

        Args:
            text: Input text

        Returns:
            Normalized text embedding
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="np"
        )

        # Run inference
        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64)
        }
        outputs = self.text_session.run(None, ort_inputs)

        # Normalize
        embedding = outputs[0][0]
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        return embedding.astype(np.float32)

    def encode_images_batch(self, images: List[Union[str, Image.Image, np.ndarray]],
                           batch_size: int = 32) -> List[np.ndarray]:
        """
        Encode multiple images in batches

        Args:
            images: List of images
            batch_size: Batch size for inference

        Returns:
            List of image embeddings
        """
        embeddings = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]

            # Preprocess batch
            batch_tensors = [self.preprocess_image(img) for img in batch]
            batch_tensor = np.concatenate(batch_tensors, axis=0)

            # Run inference
            inputs = {self.visual_session.get_inputs()[0].name: batch_tensor}
            outputs = self.visual_session.run(None, inputs)

            # Normalize
            batch_embeddings = outputs[0]
            batch_embeddings = batch_embeddings / (np.linalg.norm(batch_embeddings, axis=1, keepdims=True) + 1e-8)

            embeddings.extend(batch_embeddings)

        return embeddings

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between embeddings

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Similarity score
        """
        return float(np.dot(embedding1, embedding2))

    def ndims(self) -> int:
        """Get embedding dimensionality"""
        return 512 if "b32" in self.model_name else 512


class ONNXDINOv3Embeddings:
    """
    ONNX-based DINOv3 embeddings for efficient visual feature extraction
    """

    def __init__(self, model_name: str = "dinov2_vitb14",
                 providers: List[str] = None,
                 cache_dir: str = "~/.cache/onnx_models"):
        """
        Initialize ONNX DINOv3 model

        Args:
            model_name: Model variant (dinov2_vits14, dinov2_vitb14)
            providers: ONNX Runtime providers
            cache_dir: Directory to cache models
        """
        if providers is None:
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available_providers:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                print("✓ Using GPU acceleration (CUDA)")
            else:
                providers = ['CPUExecutionProvider']
                print("✓ Using CPU")

        self.providers = providers
        self.model_name = model_name

        # Download model if needed
        print(f"Loading {model_name} ONNX model...")
        model_paths = ONNXModelDownloader.download_model(model_name, cache_dir)

        # Load ONNX model
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            model_paths["model"],
            sess_options=sess_options,
            providers=providers
        )

        print(f"✓ Loaded {model_name} ONNX model")

    def preprocess_image(self, image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Preprocess image for DINOv3

        Args:
            image: Input image

        Returns:
            Preprocessed image tensor
        """
        # Convert to PIL Image
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image).convert('RGB')
        else:
            img = image.convert('RGB')

        # Resize (DINOv3 uses 224x224)
        img = img.resize((224, 224), Image.BICUBIC)
        img_array = np.array(img).astype(np.float32) / 255.0

        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std

        # Change to CHW format and add batch dimension
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)

        return img_array.astype(np.float32)

    def encode_image(self, image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Encode image to embedding

        Args:
            image: Input image

        Returns:
            Normalized image embedding
        """
        img_tensor = self.preprocess_image(image)

        # Run inference
        inputs = {self.session.get_inputs()[0].name: img_tensor}
        outputs = self.session.run(None, inputs)

        # Get CLS token embedding (first token)
        embedding = outputs[0][0, 0]  # [batch, seq_len, hidden] -> [hidden]

        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        return embedding.astype(np.float32)

    def encode_images_batch(self, images: List[Union[str, Image.Image, np.ndarray]],
                           batch_size: int = 32) -> List[np.ndarray]:
        """
        Encode multiple images in batches

        Args:
            images: List of images
            batch_size: Batch size for inference

        Returns:
            List of image embeddings
        """
        from tqdm import tqdm

        embeddings = []

        for i in tqdm(range(0, len(images), batch_size), desc="Extracting DINOv3 features"):
            batch = images[i:i + batch_size]

            # Preprocess batch
            batch_tensors = [self.preprocess_image(img) for img in batch]
            batch_tensor = np.concatenate(batch_tensors, axis=0)

            # Run inference
            inputs = {self.session.get_inputs()[0].name: batch_tensor}
            outputs = self.session.run(None, inputs)

            # Get CLS token embeddings
            batch_embeddings = outputs[0][:, 0, :]  # [batch, seq_len, hidden] -> [batch, hidden]

            # Normalize
            batch_embeddings = batch_embeddings / (np.linalg.norm(batch_embeddings, axis=1, keepdims=True) + 1e-8)

            embeddings.extend(batch_embeddings)

        return embeddings

    def extract_dense_features(self, image: Union[str, Image.Image]) -> Tuple[np.ndarray, int]:
        """
        Extract dense patch-level features

        Args:
            image: Input image

        Returns:
            Tuple of (features_grid, grid_size)
        """
        img_tensor = self.preprocess_image(image)

        # Run inference
        inputs = {self.session.get_inputs()[0].name: img_tensor}
        outputs = self.session.run(None, inputs)

        # Get patch tokens (exclude CLS token)
        patch_features = outputs[0][0, 1:, :]  # [seq_len-1, hidden]

        # Reshape to spatial grid
        num_patches = patch_features.shape[0]
        grid_size = int(np.sqrt(num_patches))

        patch_features = patch_features.reshape(grid_size, grid_size, -1)

        # Normalize
        patch_features = patch_features / (np.linalg.norm(patch_features, axis=-1, keepdims=True) + 1e-8)

        return patch_features, grid_size

    def ndims(self) -> int:
        """Get embedding dimensionality"""
        return 384 if "vits14" in self.model_name else 768


def test_onnx_models():
    """Test ONNX models"""
    print("Testing ONNX models...\n")

    # Create dummy image
    dummy_img = Image.new('RGB', (224, 224), color='red')

    # Test CLIP
    print("1. Testing CLIP ONNX...")
    clip = ONNXCLIPEmbeddings(model_name="clip_vit_b32")

    img_emb = clip.encode_image(dummy_img)
    print(f"   Image embedding shape: {img_emb.shape}")

    text_emb = clip.encode_text("a red image")
    print(f"   Text embedding shape: {text_emb.shape}")

    similarity = clip.compute_similarity(img_emb, text_emb)
    print(f"   Similarity: {similarity:.4f}")

    # Test DINOv3
    print("\n2. Testing DINOv3 ONNX...")
    dino = ONNXDINOv3Embeddings(model_name="dinov2_vitb14")

    img_emb = dino.encode_image(dummy_img)
    print(f"   Image embedding shape: {img_emb.shape}")

    dense_features, grid_size = dino.extract_dense_features(dummy_img)
    print(f"   Dense features shape: {dense_features.shape}")
    print(f"   Grid size: {grid_size}x{grid_size}")

    print("\n✓ All ONNX models working!")


if __name__ == "__main__":
    test_onnx_models()
