import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Streamlit app configuration
st.set_page_config(page_title="SQL Query Generator", page_icon="🗃️", layout="centered")

# Title and description
st.title("SQL Query Generator")
st.write("Enter a natural language prompt to generate a SQL query using a Hugging Face model.")

# Input for Hugging Face API token
hf_api_token = st.text_input("Hugging Face API Token", type="password", help="Enter your Hugging Face API token")

# Input for the prompt
prompt = st.text_area("Enter your prompt", placeholder="e.g., Show me all employees with salary greater than 50000", height=100)

# Model ID
MODEL_ID = "taneeshk12/t5-prompt_to_SQL"  # Your model ID

# Function to generate SQL query
def generate_sql_query(question, model, tokenizer, device):
    try:
        # Prepare the input
        input_text = "question: " + question
        input_encodings = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)

        # Generate the output sequence
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation for inference
            output_sequences = model.generate(
                input_ids=input_encodings['input_ids'],
                attention_mask=input_encodings['attention_mask'],
                max_length=512,  # Maximum length of the generated sequence
                num_beams=5,     # Use beam search for better results
                early_stopping=True  # Stop generation when all beam hypotheses have finished
            )

        # Decode the generated sequence
        predicted_query = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        return predicted_query
    except Exception as e:
        logger.error(f"Error in generate_sql_query: {str(e)}")
        return f"Error generating SQL query: {str(e)}"

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer(api_token):
    try:
        logger.info("Starting model and tokenizer loading...")
        if not api_token:
            raise ValueError("Hugging Face API token is empty")
        
        os.environ["HF_TOKEN"] = api_token
        logger.info(f"Loading tokenizer for model: {MODEL_ID}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_auth_token=api_token)
        logger.info(f"Loading model: {MODEL_ID}")
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, use_auth_token=api_token)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        model = model.to(device)
        logger.info("Model and tokenizer loaded successfully")
        return model, tokenizer, device, None
    except Exception as e:
        logger.error(f"Error loading model or tokenizer: {str(e)}")
        return None, None, None, f"Error loading model or tokenizer: {str(e)}"

# Button to generate SQL query
if st.button("Generate SQL Query"):
    if not hf_api_token:
        st.error("Please provide a valid Hugging Face API token.")
    elif not prompt:
        st.error("Please enter a prompt.")
    else:
        with st.spinner("Loading model and generating SQL query..."):
            model, tokenizer, device, error = load_model_and_tokenizer(hf_api_token)
            if error:
                st.error(error)
            else:
                sql_query = generate_sql_query(prompt, model, tokenizer, device)
                st.subheader("Generated SQL Query")
                st.code(sql_query, language="sql")

# Instructions for the user
st.markdown("""
### Instructions
1. Enter your Hugging Face API token from [Hugging Face](https://huggingface.co/settings/tokens).
2. Ensure the model ID `taneeshk12/t5-prompt_to_SQL` is correct and accessible.
3. Enter a natural language prompt describing the SQL query you want to generate.
4. Click the "Generate SQL Query" button to see the result.
5. If you encounter errors, check the terminal for detailed logs.
""")