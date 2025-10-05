import google.generativeai as genai
import os

# Configure the Gemini API key
# Make sure to set your GOOGLE_API_KEY as an environment variable
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# Initialize the models
summary_model = genai.GenerativeModel('gemini-1.5-pro-latest')
scoring_model = genai.GenerativeModel('gemini-1.5-flash-latest')

def generate_summary(image_data: bytes = None, text_data: str = None) -> str:
    """
    Generates a diagnostic summary using Gemini 1.5 Pro.
    Accepts either image or text data, or both.
    """
    prompt_parts = []
    
    base_prompt = """
    You are a clinical AI assistant. Analyze the provided medical data and generate a structured diagnostic summary.
    The summary should be in markdown format with two sections: '### Findings' and '### Impressions'.
    - Findings: Detail the objective observations from the data.
    - Impressions: Provide a concise clinical interpretation and potential diagnosis.
    Maintain professional medical terminology.
    """
    prompt_parts.append(base_prompt)

    if image_data:
        image_part = {
            "mime_type": "image/jpeg", # Assuming JPEG, adjust if needed
            "data": image_data
        }
        prompt_parts.append(image_part)

    if text_data:
        prompt_parts.append(f"\nClinical Notes/Text Input:\n{text_data}")

    if not image_data and not text_data:
        raise ValueError("Either image or text data must be provided.")

    response = summary_model.generate_content(prompt_parts)
    return response.text

def calculate_trust_score_and_regenerate(original_summary: str, feedback: str) -> tuple[float, str]:
    """
    Uses Gemini 1.5 Flash to calculate a trust score based on feedback and
    Gemini 1.5 Pro to regenerate the summary.
    """
    # --- Step 1: Calculate Trust Score using Gemini Flash ---
    scoring_prompt = f"""
    Analyze the original AI-generated medical summary and the clinician's feedback.
    Based on the severity and nature of the correction, provide a "Trust Score" between 0 and 100 for the ORIGINAL summary.
    A score of 100 means the original summary was perfect. A score of 0 means it was completely wrong.
    Consider factual errors (e.g., wrong side of the body) as major deductions.
    Consider stylistic preferences or minor additions as small deductions.
    
    Return ONLY the numerical score. For example: 85.5

    Original Summary:
    "{original_summary}"

    Clinician Feedback:
    "{feedback}"

    Trust Score:
    """
    
    score_response = scoring_model.generate_content(scoring_prompt)
    try:
        # Clean up the response to ensure it's a float
        score_text = score_response.text.strip().replace('%', '')
        trust_score = float(score_text)
    except (ValueError, TypeError):
        print(f"Could not parse score from response: '{score_response.text}'. Defaulting to 0.")
        trust_score = 0.0 # Default score if parsing fails

    # --- Step 2: Regenerate Summary using Gemini Pro ---
    regeneration_prompt = f"""
    You are a clinical AI assistant. You previously generated a medical summary which a clinician has reviewed.
    Your task is to regenerate the summary by incorporating the clinician's feedback accurately.
    Produce a new, corrected summary in the same '### Findings' and '### Impressions' markdown format.

    Original Summary:
    "{original_summary}"

    Clinician's Feedback for Correction:
    "{feedback}"

    Regenerated Summary:
    """

    regenerated_response = summary_model.generate_content(regeneration_prompt)
    regenerated_summary = regenerated_response.text

    return trust_score, regenerated_summary
