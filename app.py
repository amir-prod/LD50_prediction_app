import streamlit as st
import chemprop
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import io

# Set page configuration
st.set_page_config(
    page_title="Toxicity Prediction App",
    page_icon="🧪",
    layout="wide"
)

# Title and description
st.title("🧪 Molecular Toxicity Prediction")
st.markdown("""
This application predicts the **rat oral acute toxicity (LD50)** of chemical compounds from their SMILES strings.
You can either upload a CSV file with SMILES or enter them manually.
""")

# Load model (cached to avoid reloading on every interaction)
@st.cache_resource
def load_chemprop_model():
    """Load the chemprop model"""
    arguments = [
        '--test_path', '/dev/null',
        '--preds_path', '/dev/null',
        '--checkpoint_dir', '/home/amir.daghighi/projects/ICE_toxicity/ICE_v4/codes/streamlit/',
        '--features_generator', 'rdkit_2d_normalized',
        '--no_features_scaling'
    ]
    
    args = chemprop.args.PredictArgs().parse_args(arguments)
    model_objects = chemprop.train.load_model(args=args)
    
    return args, model_objects

# Function to convert -logLD50 to LD50
def convert_to_ld50(preds, smiles_list):
    """
    Convert predictions from -log(LD50/MW) to LD50 (mg/kg)
    
    Args:
        preds: List of predictions in -log(LD50/MW) format
        smiles_list: List of SMILES strings
    
    Returns:
        List of LD50 values in mg/kg
    """
    ld50_values = []
    
    for pred, smiles in zip(preds, smiles_list):
        # Get molecular weight
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mw = Descriptors.MolWt(mol)
            # Convert: pred = -log10(LD50/MW)
            # So: LD50 = MW * 10^(-pred)
            ld50 = mw * (10 ** (-pred[0]))
            ld50_values.append(ld50)
        else:
            ld50_values.append(None)
    
    return ld50_values

# Function to validate SMILES
def validate_smiles(smiles):
    """Check if SMILES string is valid"""
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

# Load model with progress indicator
with st.spinner('Loading model... This may take a moment.'):
    try:
        args, model_objects = load_chemprop_model()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

# Sidebar for input method selection
st.sidebar.header("Input Method")
input_method = st.sidebar.radio(
    "Choose how to input SMILES:",
    ["Manual Input", "CSV File Upload"]
)

# Initialize session state for storing SMILES
if 'smiles_list' not in st.session_state:
    st.session_state.smiles_list = []

# Main content area
smiles_list = []

if input_method == "Manual Input":
    st.header("Manual SMILES Input")
    
    # Text area for SMILES input
    smiles_input = st.text_area(
        "Enter SMILES strings (one per line):",
        height=200,
        placeholder="Example:\nCCC\nCCCC\nOCC"
    )
    
    # Submit button for manual input
    submit_button = st.button("Submit SMILES", type="secondary")
    
    if smiles_input and submit_button:
        st.session_state.smiles_list = [s.strip() for s in smiles_input.split('\n') if s.strip()]
        st.info(f"📝 {len(st.session_state.smiles_list)} SMILES string(s) entered")
    
    # Use the stored SMILES list from session state
    smiles_list = st.session_state.smiles_list

else:  # CSV File Upload
    st.header("CSV File Upload")
    
    # Clear manual input session state when switching to CSV mode
    st.session_state.smiles_list = []
    
    st.markdown("""
    **CSV Format Requirements:**
    - Must contain a column with SMILES strings
    - You'll be able to select which column contains the SMILES
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Display preview
            st.subheader("File Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Column selection
            smiles_column = st.selectbox(
                "Select the column containing SMILES:",
                options=df.columns.tolist()
            )
            
            if smiles_column:
                smiles_list = df[smiles_column].astype(str).tolist()
                st.info(f"📊 {len(smiles_list)} SMILES string(s) loaded from column '{smiles_column}'")
                
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")

# Prediction section
if smiles_list:
    st.header("Predictions")
    
    # Validate SMILES
    valid_smiles = []
    invalid_smiles = []
    
    for smiles in smiles_list:
        if validate_smiles(smiles):
            valid_smiles.append(smiles)
        else:
            invalid_smiles.append(smiles)
    
    # Show validation results
    if invalid_smiles:
        st.warning(f"⚠️ {len(invalid_smiles)} invalid SMILES string(s) found and will be skipped:")
        with st.expander("Show invalid SMILES"):
            for smiles in invalid_smiles:
                st.code(smiles)
    
    if valid_smiles:
        st.info(f"✅ {len(valid_smiles)} valid SMILES string(s) ready for prediction")
        
        # Predict button
        if st.button("🔬 Predict Toxicity", type="primary"):
            with st.spinner('Making predictions...'):
                try:
                    # Format SMILES for chemprop (needs list of lists)
                    smiles_formatted = [[s] for s in valid_smiles]
                    
                    # Make predictions
                    preds = chemprop.train.make_predictions(
                        args=args,
                        smiles=smiles_formatted,
                        model_objects=model_objects
                    )
                    
                    # Convert to LD50
                    ld50_values = convert_to_ld50(preds, valid_smiles)
                    
                    # Create results dataframe
                    results_df = pd.DataFrame({
                        'SMILES': valid_smiles,
                        '-log(LD50/MW)': [round(pred[0], 4) for pred in preds],
                        'LD50 (mg/kg)': [round(ld50, 1) for ld50 in ld50_values]
                    })
                    
                    # Add molecular weight
                    results_df['Molecular Weight (g/mol)'] = [
                        Descriptors.MolWt(Chem.MolFromSmiles(s)) for s in valid_smiles
                    ]
                    
                    # Reorder columns
                    results_df = results_df[['SMILES', 'Molecular Weight (g/mol)', '-log(LD50/MW)', 'LD50 (mg/kg)']]
                    
                    # Display results
                    st.success("✅ Predictions completed!")
                    
                    # # Summary statistics
                    # col1, col2, col3 = st.columns(3)
                    # with col1:
                    #     st.metric("Average LD50", f"{results_df['LD50 (mg/kg)'].mean():.2f} mg/kg")
                    # with col2:
                    #     st.metric("Min LD50", f"{results_df['LD50 (mg/kg)'].min():.2f} mg/kg")
                    # with col3:
                    #     st.metric("Max LD50", f"{results_df['LD50 (mg/kg)'].max():.2f} mg/kg")
                    
                    # Display full results table
                    st.subheader("Detailed Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="toxicity_predictions.csv",
                        mime="text/csv"
                    )
                    
                    # Interpretation guide
                    st.subheader("📖 Interpretation Guide")
                    st.markdown("""
                    - **LD50**: Lethal Dose 50 - the dose that causes death in 50% of test subjects
                    - **Lower LD50 values** indicate **higher toxicity** (less compound needed to be lethal)
                    - **Higher LD50 values** indicate **lower toxicity** (more compound needed to be lethal)
                    
                    **Toxicity Classification (approximate):**
                    - Extremely toxic: LD50 < 5 mg/kg
                    - Highly toxic: LD50 5-50 mg/kg
                    - Moderately toxic: LD50 50-500 mg/kg
                    - Slightly toxic: LD50 500-5000 mg/kg
                    - Practically non-toxic: LD50 > 5000 mg/kg
                    """)
                    
                except Exception as e:
                    st.error(f"❌ Error making predictions: {str(e)}")
                    st.exception(e)
    else:
        st.warning("⚠️ No valid SMILES strings to predict")
else:
    st.info("👆 Please enter SMILES strings or upload a CSV file to begin")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Streamlit and Chemprop | Toxicity Prediction Model</p>
</div>
""", unsafe_allow_html=True)

