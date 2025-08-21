"""
Configuration settings for Procurement Optimization AI
Centralized configuration management
"""

import os
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    url: str = "sqlite:///procurement.db"
    echo: bool = False
    pool_size: int = 10
    max_overflow: int = 20


@dataclass
class LLMConfig:
    """LLM configuration settings"""
    default_model_type: str = "ollama"
    default_model_name: str = "mistral:7b"
    temperature: float = 0.1
    max_tokens: int = 500
    top_p: float = 0.9
    repeat_penalty: float = 1.1


@dataclass
class AnalysisConfig:
    """Analysis configuration settings"""
    forecast_days: int = 30
    confidence_level: float = 0.95
    min_data_points: int = 10
    max_forecast_period: int = 365


@dataclass
class VisualizationConfig:
    """Visualization configuration settings"""
    theme: str = "plotly_white"
    color_scheme: str = "viridis"
    chart_height: int = 500
    chart_width: int = 800


@dataclass
class AppConfig:
    """Application configuration settings"""
    debug: bool = False
    log_level: str = "INFO"
    cache_timeout: int = 3600
    max_file_size: int = 50 * 1024 * 1024  # 50MB


class Config:
    """Main configuration class"""
    
    def __init__(self):
        """Initialize configuration with default values"""
        self.database = DatabaseConfig()
        self.llm = LLMConfig()
        self.analysis = AnalysisConfig()
        self.visualization = VisualizationConfig()
        self.app = AppConfig()
        
        # Load environment variables
        self._load_env_vars()
    
    def _load_env_vars(self):
        """Load configuration from environment variables"""
        
        # Database settings
        if os.getenv("DATABASE_URL"):
            self.database.url = os.getenv("DATABASE_URL")
        
        if os.getenv("DATABASE_ECHO"):
            self.database.echo = os.getenv("DATABASE_ECHO").lower() == "true"
        
        # LLM settings
        if os.getenv("LLM_TYPE"):
            self.llm.default_model_type = os.getenv("LLM_TYPE")
        
        if os.getenv("LLM_MODEL"):
            self.llm.default_model_name = os.getenv("LLM_MODEL")
        
        if os.getenv("LLM_TEMPERATURE"):
            self.llm.temperature = float(os.getenv("LLM_TEMPERATURE"))
        
        # Analysis settings
        if os.getenv("FORECAST_DAYS"):
            self.analysis.forecast_days = int(os.getenv("FORECAST_DAYS"))
        
        if os.getenv("CONFIDENCE_LEVEL"):
            self.analysis.confidence_level = float(os.getenv("CONFIDENCE_LEVEL"))
        
        # App settings
        if os.getenv("DEBUG"):
            self.app.debug = os.getenv("DEBUG").lower() == "true"
        
        if os.getenv("LOG_LEVEL"):
            self.app.log_level = os.getenv("LOG_LEVEL")
    
    def get_database_url(self) -> str:
        """Get database URL"""
        return self.database.url
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration"""
        return {
            "model_type": self.llm.default_model_type,
            "model_name": self.llm.default_model_name,
            "temperature": self.llm.temperature,
            "max_tokens": self.llm.max_tokens,
            "top_p": self.llm.top_p,
            "repeat_penalty": self.llm.repeat_penalty
        }
    
    def get_analysis_config(self) -> Dict[str, Any]:
        """Get analysis configuration"""
        return {
            "forecast_days": self.analysis.forecast_days,
            "confidence_level": self.analysis.confidence_level,
            "min_data_points": self.analysis.min_data_points,
            "max_forecast_period": self.analysis.max_forecast_period
        }
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """Get visualization configuration"""
        return {
            "theme": self.visualization.theme,
            "color_scheme": self.visualization.color_scheme,
            "chart_height": self.visualization.chart_height,
            "chart_width": self.visualization.chart_width
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "database": {
                "url": self.database.url,
                "echo": self.database.echo,
                "pool_size": self.database.pool_size,
                "max_overflow": self.database.max_overflow
            },
            "llm": {
                "default_model_type": self.llm.default_model_type,
                "default_model_name": self.llm.default_model_name,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
                "top_p": self.llm.top_p,
                "repeat_penalty": self.llm.repeat_penalty
            },
            "analysis": {
                "forecast_days": self.analysis.forecast_days,
                "confidence_level": self.analysis.confidence_level,
                "min_data_points": self.analysis.min_data_points,
                "max_forecast_period": self.analysis.max_forecast_period
            },
            "visualization": {
                "theme": self.visualization.theme,
                "color_scheme": self.visualization.color_scheme,
                "chart_height": self.visualization.chart_height,
                "chart_width": self.visualization.chart_width
            },
            "app": {
                "debug": self.app.debug,
                "log_level": self.app.log_level,
                "cache_timeout": self.app.cache_timeout,
                "max_file_size": self.app.max_file_size
            }
        }


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get global configuration instance"""
    return config


def update_config(updates: Dict[str, Any]):
    """Update configuration with new values"""
    global config
    
    for section, values in updates.items():
        if hasattr(config, section):
            section_config = getattr(config, section)
            for key, value in values.items():
                if hasattr(section_config, key):
                    setattr(section_config, key, value)


def create_env_template() -> str:
    """Create environment variables template"""
    template = """# Procurement Optimization AI - Environment Variables

# Database Configuration
DATABASE_URL=sqlite:///procurement.db
DATABASE_ECHO=false

# LLM Configuration
LLM_TYPE=ollama
LLM_MODEL=mistral:7b
LLM_TEMPERATURE=0.1

# Analysis Configuration
FORECAST_DAYS=30
CONFIDENCE_LEVEL=0.95

# Application Configuration
DEBUG=false
LOG_LEVEL=INFO

# Optional: PostgreSQL Database
# DATABASE_URL=postgresql://user:password@localhost:5432/procurement

# Optional: Hugging Face Models
# LLM_TYPE=huggingface
# LLM_MODEL=microsoft/DialoGPT-medium
"""
    return template


def save_env_template(filename: str = ".env.template"):
    """Save environment variables template to file"""
    template = create_env_template()
    with open(filename, 'w') as f:
        f.write(template)
    print(f"Environment template saved to {filename}")


if __name__ == "__main__":
    # Print current configuration
    print("Current Configuration:")
    print(config.to_dict())
    
    # Create environment template
    save_env_template()
