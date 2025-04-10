import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import time
import os
import json

class ProgressTracker:
    """Class for tracking progress of long-running operations."""
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        """
        Initialize the progress tracker.
        
        Args:
            total_steps: Total number of steps in the operation
            description: Description of the operation
        """
        self.total_steps = max(1, total_steps)  # Ensure at least 1 step
        self.current_step = 0
        self.description = description
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        
    def update(self, step: int = None, description: str = None) -> None:
        """
        Update the progress tracker.
        
        Args:
            step: Current step (if None, increment by 1)
            description: New description (if provided)
        """
        if step is not None:
            self.current_step = min(step, self.total_steps)
        else:
            self.current_step = min(self.current_step + 1, self.total_steps)
        
        if description:
            self.description = description
            
        # Update progress bar
        progress = self.current_step / self.total_steps
        self.progress_bar.progress(progress)
        
        # Update status text
        self.status_text.text(f"{self.description}: {int(progress * 100)}%")
        
    def complete(self, message: str = "Complete!") -> None:
        """
        Mark the operation as complete.
        
        Args:
            message: Completion message to display
        """
        self.progress_bar.progress(1.0)
        self.status_text.text(message)
        time.sleep(1)  # Brief pause to show completion
        self.progress_bar.empty()
        self.status_text.empty()

class DataVisualizer:
    """Class for creating visualizations of search results."""
    
    @staticmethod
    def create_source_distribution_chart(source_locations: Dict[str, int]) -> None:
        """
        Create a pie chart showing distribution of results by source.
        
        Args:
            source_locations: Dictionary mapping source locations to result counts
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data
        labels = list(source_locations.keys())
        values = list(source_locations.values())
        
        # Create pie chart
        ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title('Distribution of Results by Source')
        
        # Display in Streamlit
        st.pyplot(fig)
    
    @staticmethod
    def create_query_effectiveness_chart(analysis: Dict[str, Any]) -> None:
        """
        Create a bar chart showing effectiveness of different queries.
        
        Args:
            analysis: Analysis dictionary from EnhancedQueryAgent.analyze_results()
        """
        if 'results_by_query' not in analysis:
            return
            
        # Prepare data
        queries = list(analysis['results_by_query'].keys())
        result_counts = list(analysis['results_by_query'].values())
        
        # Create DataFrame
        df = pd.DataFrame({
            'Query': queries,
            'Result Count': result_counts
        })
        
        # Sort by result count
        df = df.sort_values('Result Count', ascending=False)
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Result Count', y='Query', data=df, ax=ax)
        plt.title('Number of Results by Query')
        plt.tight_layout()
        
        # Display in Streamlit
        st.pyplot(fig)

class ConfigManager:
    """Class for managing application configuration."""
    
    def __init__(self, config_path: str = "./config.json"):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.default_config = {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "max_reformulations": 3,
            "use_cache": True,
            "gpt_models": {
                "embeddings": "text-embedding-ada-002",
                "vision": "gpt-4-vision-preview",
                "completion": "gpt-4-turbo"
            }
        }
        
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file or create default.
        
        Returns:
            Configuration dictionary
        """
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except Exception:
                return self.default_config.copy()
        else:
            return self.default_config.copy()
    
    def save_config(self) -> None:
        """Save configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            st.error(f"Error saving configuration: {str(e)}")
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
        """
        for key, value in updates.items():
            if isinstance(value, dict) and key in self.config and isinstance(self.config[key], dict):
                # Merge nested dictionaries
                self.config[key].update(value)
            else:
                # Replace value
                self.config[key] = value
        
        self.save_config()
