# Data Synthesizer for ML Prototyping

**Generate fake datasets in seconds for testing your ML models!**

Create custom synthetic data with controllable properties (noise, trends, imbalance) and start prototyping immediately.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/data-synthesizer.git
cd data-synthesizer

# Install dependencies
pip install -r requirements.txt
```

## 📊 What Can You Generate?
 **Time Series**: Stock prices, sensor readings, metrics, etc. 
 **Classification**: Spam detection, image labels, categories, etc. 
 **Regression**: Price prediction, forecasting, etc.

### 🎯 Why Use This?
✅ **Educational**      --> Learn ML concepts with clean examples  
✅ **Fast prototyping** --> Generate data in 3 lines of code  
✅ **Reproducible**     --> Same seed = same data every time  
✅ **Configurable**     --> Control noise, trends, class balance  
✅ **No setup**         --> Just pip/conda install 


### 💡 Examples

Check the `examples/` folder for a complete working example.


### 📁 Project Structure

```
data-synthesizer/
├── src/                       # Main package
│   ├── __init__.py
│   ├── timeseries.py          # Time series generator
│   ├── tabular.py             # Classification/regression generator
│   └── utils.py               # Utility functions
├── example/ 
|   └── ex01.py                # Example outputs
├── requirements.txt           # Dependencies
└── README.md                  # This file
```