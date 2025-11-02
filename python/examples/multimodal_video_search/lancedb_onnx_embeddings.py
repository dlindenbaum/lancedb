"""
LanceDB-compatible ONNX embedding functions

Adapters to integrate ONNX models with LanceDB's embedding system.
"""

from typing import List, Union
import numpy as np
from PIL import Image
import pyarrow as pa

from onnx_embeddings import ONNXCLIPEmbeddings


class LanceDBONNXCLIPEmbeddings:
    """
    LanceDB-compatible CLIP ONNX embedding function

    This adapter makes ONNX CLIP models work seamlessly with LanceDB.
    """

    def __init__(self, model_name: str = "clip_vit_b32",
                 providers: List[str] = None,
                 cache_dir: str = "~/.cache/onnx_models"):
        """
        Initialize ONNX CLIP embeddings for LanceDB

        Args:
            model_name: CLIP model variant
            providers: ONNX Runtime providers
            cache_dir: Model cache directory
        """
        self.onnx_model = ONNXCLIPEmbeddings(
            model_name=model_name,
            providers=providers,
            cache_dir=cache_dir
        )
        self.model_name = model_name
        self._ndims = None

    def ndims(self) -> int:
        """Get embedding dimensionality"""
        if self._ndims is None:
            self._ndims = self.onnx_model.ndims()
        return self._ndims

    def compute_query_embeddings(
        self, query: Union[str, Image.Image], *args, **kwargs
    ) -> List[np.ndarray]:
        """
        Compute embeddings for a query (text or image)

        Args:
            query: Text string or PIL Image

        Returns:
            List containing the query embedding
        """
        if isinstance(query, str):
            embedding = self.onnx_model.encode_text(query)
        elif isinstance(query, Image.Image):
            embedding = self.onnx_model.encode_image(query)
        else:
            raise TypeError(f"Unsupported query type: {type(query)}")

        return [embedding]

    def compute_source_embeddings(
        self, images: Union[List, pa.Array, pa.ChunkedArray], *args, **kwargs
    ) -> List[np.ndarray]:
        """
        Compute embeddings for source images

        Args:
            images: List of image paths or PIL Images

        Returns:
            List of image embeddings
        """
        # Sanitize input
        if isinstance(images, pa.Array):
            images = images.to_pylist()
        elif isinstance(images, pa.ChunkedArray):
            images = images.combine_chunks().to_pylist()
        elif not isinstance(images, list):
            images = [images]

        # Batch encode
        embeddings = self.onnx_model.encode_images_batch(images, batch_size=32)

        return embeddings

    def generate_text_embeddings(self, text: str) -> np.ndarray:
        """Generate embedding for text query"""
        return self.onnx_model.encode_text(text)

    def generate_image_embedding(
        self, image: Union[str, bytes, Image.Image]
    ) -> np.ndarray:
        """
        Generate embedding for a single image

        Args:
            image: Image path, bytes, or PIL Image

        Returns:
            Image embedding
        """
        # Handle bytes
        if isinstance(image, bytes):
            import io
            image = Image.open(io.BytesIO(image))

        return self.onnx_model.encode_image(image)


def create_onnx_clip_embeddings(
    model_name: str = "clip_vit_b32",
    device: str = "cuda"
) -> LanceDBONNXCLIPEmbeddings:
    """
    Factory function to create ONNX CLIP embeddings

    Args:
        model_name: CLIP model variant
        device: "cuda" or "cpu"

    Returns:
        LanceDB-compatible ONNX CLIP embedding function
    """
    import onnxruntime as ort

    # Determine providers based on device
    if device == "cuda":
        available = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            print("⚠ CUDA requested but not available, falling back to CPU")
            providers = ['CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']

    return LanceDBONNXCLIPEmbeddings(
        model_name=model_name,
        providers=providers
    )
