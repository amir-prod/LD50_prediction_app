# Molecular Toxicity Prediction Web App

A Streamlit web application for predicting LD50 toxicity of chemical compounds from SMILES strings using a trained Chemprop model.

## Features

- 🧪 **Toxicity Prediction**: Predicts LD50 values from SMILES strings
- 📝 **Manual Input**: Enter SMILES strings directly in the web interface
- 📊 **CSV Upload**: Upload a CSV file containing multiple SMILES strings
- ✅ **SMILES Validation**: Automatically validates input SMILES strings
- 📥 **Export Results**: Download predictions as CSV


## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Navigate to the streamlit directory:
```bash
cd /your/project/directory
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. The app will open in your default web browser (usually at http://localhost:8501)

## Input Methods

### Manual Input
- Enter one SMILES string per line in the text area
- Example:
  ```
  CCC
  CCCC
  OCC
  ```

### CSV Upload
- Upload a CSV file containing SMILES strings
- Select the column that contains the SMILES data
- The app will automatically process all valid SMILES

## Output

The application provides:
- **-log(LD50/MW)**: Raw model prediction
- **LD50 (mg/kg)**: Converted toxicity value
- **Molecular Weight**: Calculated molecular weight
- **Summary Statistics**: Average, min, and max LD50 values
- **Downloadable CSV**: Results can be exported for further analysis

## Understanding LD50 Values

- **Lower LD50** = **Higher Toxicity** (more dangerous)
- **Higher LD50** = **Lower Toxicity** (less dangerous)

### Toxicity Classification according to OECD:
- Extremely toxic: LD50 < 5 mg/kg
- Highly toxic: LD50 5-50 mg/kg
- Moderately toxic: LD50 50-500 mg/kg
- Slightly toxic: LD50 500-5000 mg/kg
- Practically non-toxic: LD50 > 5000 mg/kg

## Model Information

The predictions are made using a Chemprop graph neural network model with:
- RDKit 2D normalized features
- Trained on ICE v4 toxicity data
- Predicts -log(LD50/MW) which is converted to LD50 (mg/kg)

## Files

- `app.py`: Main Streamlit application
- `model.pt`: Trained Chemprop model checkpoint
- `requirements.txt`: Python dependencies
- `tmp.ipynb`: Original notebook with model prediction code

