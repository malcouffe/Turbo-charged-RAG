import os
import json
import hashlib
from typing import Dict, Optional, Any
from datetime import datetime

class ImageAnalysisCache:
    """
    Cache for storing image analysis results to avoid redundant API calls to GPT-4 Vision.
    """
    
    def __init__(self, cache_dir: str = "./.cache", cache_expiry_days: int = 30):
        """
        Initialize the image analysis cache.
        
        Args:
            cache_dir: Directory to store cache files
            cache_expiry_days: Number of days after which cache entries expire
        """
        self.cache_dir = cache_dir
        self.cache_expiry_days = cache_expiry_days
        self.cache_file = os.path.join(cache_dir, "image_analysis_cache.json")
        self.cache = {}
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load cache from file if it exists
        self.load_cache()
    
    def load_cache(self) -> None:
        """Load the cache from the cache file."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    self.cache = json.load(f)
            except Exception as e:
                print(f"Error loading cache: {str(e)}")
                self.cache = {}
    
    def save_cache(self) -> None:
        """Save the cache to the cache file."""
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self.cache, f)
        except Exception as e:
            print(f"Error saving cache: {str(e)}")
    
    def get_image_hash(self, image_path: str) -> str:
        """
        Calculate a hash for the image to use as cache key.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Hash string for the image
        """
        try:
            with open(image_path, "rb") as f:
                image_data = f.read()
                return hashlib.md5(image_data).hexdigest()
        except Exception:
            # If we can't read the image, use the path as key
            return hashlib.md5(image_path.encode()).hexdigest()
    
    def get_analysis(self, image_path: str) -> Optional[str]:
        """
        Get cached analysis for an image if available.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Cached analysis text or None if not found
        """
        image_hash = self.get_image_hash(image_path)
        
        if image_hash in self.cache:
            entry = self.cache[image_hash]
            
            # Check if cache entry has expired
            timestamp = entry.get("timestamp", 0)
            current_time = datetime.now().timestamp()
            if (current_time - timestamp) > (self.cache_expiry_days * 86400):
                # Cache expired
                del self.cache[image_hash]
                self.save_cache()
                return None
            
            return entry.get("analysis")
        
        return None
    
    def add_analysis(self, image_path: str, analysis: str) -> None:
        """
        Add image analysis to cache.
        
        Args:
            image_path: Path to the image file
            analysis: Analysis text to cache
        """
        image_hash = self.get_image_hash(image_path)
        
        self.cache[image_hash] = {
            "timestamp": datetime.now().timestamp(),
            "analysis": analysis,
            "path": image_path
        }
        
        self.save_cache()
    
    def clear_cache(self) -> None:
        """Clear the entire cache."""
        self.cache = {}
        self.save_cache()
