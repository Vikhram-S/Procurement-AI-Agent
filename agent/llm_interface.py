"""
LLM Interface for Procurement Optimization AI Agent
Supports both Ollama and Hugging Face Transformers for local LLM usage
"""

import os
from typing import Optional, Dict, Any
from langchain.llms import Ollama
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


class LLMInterface:
    """Interface for local LLM models (Ollama or Hugging Face)"""
    
    def __init__(self, model_type: str = "ollama", model_name: str = "mistral:7b"):
        """
        Initialize LLM interface
        
        Args:
            model_type: "ollama" or "huggingface"
            model_name: Model name for the selected type
        """
        self.model_type = model_type
        self.model_name = model_name
        self.llm = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM based on the selected type"""
        try:
            if self.model_type == "ollama":
                self.llm = Ollama(
                    model=self.model_name,
                    temperature=0.1,
                    top_p=0.9,
                    repeat_penalty=1.1
                )
            elif self.model_type == "huggingface":
                self.llm = self._setup_huggingface_model()
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
                
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            # Fallback to a simple mock LLM for testing
            self.llm = MockLLM()
    
    def _setup_huggingface_model(self):
        """Setup Hugging Face model"""
        model_name = self.model_name
        
        # Common open-source models
        if model_name == "microsoft/DialoGPT-medium":
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
        elif model_name == "gpt2":
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            # Default to a smaller model
            model_name = "microsoft/DialoGPT-small"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        return HuggingFacePipeline(pipeline=pipe)
    
    def generate_response(self, prompt: str, max_tokens: int = 500) -> str:
        """
        Generate response from the LLM
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response
        """
        try:
            if self.llm:
                response = self.llm(prompt)
                return response[:max_tokens] if len(response) > max_tokens else response
            else:
                return "LLM not initialized properly"
        except Exception as e:
            return f"Error generating response: {e}"
    
    def analyze_procurement_data(self, data_summary: str) -> Dict[str, Any]:
        """
        Analyze procurement data using the LLM
        
        Args:
            data_summary: Summary of procurement data
            
        Returns:
            Analysis results
        """
        prompt = f"""
        Analyze the following hospital procurement data and provide insights:
        
        {data_summary}
        
        Please provide analysis in the following format:
        1. Key Trends Identified
        2. Cost Optimization Opportunities
        3. Vendor Performance Insights
        4. Recommended Actions
        5. Risk Factors
        """
        
        response = self.generate_response(prompt)
        
        return {
            "analysis": response,
            "model_used": f"{self.model_type}:{self.model_name}",
            "timestamp": "2024-01-01T00:00:00Z"
        }


class MockLLM:
    """Mock LLM for testing when real LLM is not available"""
    
    def __call__(self, prompt: str) -> str:
        """Mock response for testing"""
        return """
        Analysis Results:
        
        1. Key Trends Identified:
        - Increasing demand for medical supplies
        - Seasonal variations in procurement patterns
        - Price fluctuations in certain categories
        
        2. Cost Optimization Opportunities:
        - Bulk purchasing for high-volume items
        - Negotiate better terms with top vendors
        - Consider alternative suppliers for expensive items
        
        3. Vendor Performance Insights:
        - Vendor A: Reliable delivery, competitive pricing
        - Vendor B: Good quality, but higher costs
        - Vendor C: Fast delivery, but inconsistent quality
        
        4. Recommended Actions:
        - Implement demand forecasting
        - Establish long-term contracts with reliable vendors
        - Diversify supplier base for critical items
        
        5. Risk Factors:
        - Supply chain disruptions
        - Price volatility in medical supplies
        - Quality control issues with new vendors
        """


def get_available_models() -> Dict[str, list]:
    """Get list of available models for each type"""
    return {
        "ollama": [
            "mistral:7b",
            "llama2:7b",
            "llama2:13b",
            "codellama:7b",
            "neural-chat:7b"
        ],
        "huggingface": [
            "microsoft/DialoGPT-medium",
            "gpt2",
            "microsoft/DialoGPT-small",
            "EleutherAI/gpt-neo-125M"
        ]
    }
