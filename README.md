# Procurement Optimization AI Agent

An intelligent AI agent that autonomously analyzes hospital purchase orders and vendor pricing to predict optimal procurement strategies for cost and availability optimization.

## Features

- **Autonomous Analysis**: AI agent analyzes purchase orders and vendor data
- **Cost Optimization**: Predicts optimal procurement strategies for cost reduction
- **Availability Forecasting**: Forecasts demand and availability patterns
- **Vendor Analysis**: Evaluates vendor performance and pricing
- **Interactive Dashboard**: Streamlit-based user interface
- **Open Source**: Uses local LLMs (Ollama/Mistral-7B) without paid APIs

## Tech Stack

- **Agentic AI**: LangChain + LangGraph
- **LLM**: Ollama (Mistral-7B/LLaMA-3) or Hugging Face Transformers
- **Data Handling**: Pandas + SQLAlchemy
- **Analytics**: Scikit-learn + XGBoost
- **UI**: Streamlit
- **Visualization**: Plotly + Matplotlib

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd procurement-optimization-ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Ollama (for local LLM):
```bash
# Download from https://ollama.ai/
# Then run:
ollama pull mistral:7b
```

4. Run the application:
```bash
streamlit run app.py
```

## Project Structure

```
├── app.py                          # Main Streamlit application
├── agent/
│   ├── __init__.py
│   ├── procurement_agent.py        # Main AI agent logic
│   ├── llm_interface.py           # LLM interface (Ollama/HuggingFace)
│   └── graph_workflow.py          # LangGraph workflow
├── data/
│   ├── __init__.py
│   ├── database.py                # SQLAlchemy database models
│   ├── sample_data.py             # Sample hospital data generator
│   └── data_processor.py          # Data processing utilities
├── models/
│   ├── __init__.py
│   ├── demand_forecaster.py       # Demand forecasting models
│   ├── cost_optimizer.py          # Cost optimization algorithms
│   └── vendor_analyzer.py         # Vendor analysis models
├── utils/
│   ├── __init__.py
│   ├── visualization.py           # Plotting utilities
│   └── config.py                  # Configuration settings
└── requirements.txt
```

## Usage

1. **Start the Application**: Run `streamlit run app.py`
2. **Upload Data**: Upload hospital purchase orders and vendor data
3. **Configure Agent**: Set analysis parameters and preferences
4. **Run Analysis**: Let the AI agent analyze and provide recommendations
5. **View Results**: Explore interactive visualizations and reports

## Key Components

### AI Agent (LangChain + LangGraph)
- Autonomous analysis of procurement data
- Multi-step reasoning workflow
- Cost-benefit analysis
- Vendor evaluation

### Demand Forecasting (Scikit-learn + XGBoost)
- Time series analysis
- Seasonal pattern detection
- Demand prediction models
- Inventory optimization

### Cost Optimization
- Vendor price comparison
- Bulk purchase optimization
- Contract negotiation insights
- Budget allocation strategies

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Contact:
for any inquiries : vikhrams@saveetha.ac.in
