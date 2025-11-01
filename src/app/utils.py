from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO

def create_pdf_from_summary(summary_text: str) -> bytes:
    """
    Creates a simple PDF document from a given text summary.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Simple replacement for markdown-like headers for PDF
    story_text = summary_text.replace('###', '').replace('\n', '<br/>')
    
    story = [Paragraph(story_text, styles['Normal'])]
    
    doc.build(story)
    
    buffer.seek(0)
    return buffer.getvalue()

def load_json_file(file_path: str):
    """
    Loads a JSON file and returns its content.
    """
    import json
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

import json

def get_results(filename):

  # Load your JSON file
  with open(filename, "r", encoding="utf-8") as f:
      data = json.load(f)

  # Initialize empty variables
  user_prompt = ""
  assistant_response = ""

  # Loop through the list in data["output"]
  for item in data.get("output", []):
      role = item.get("role")
      content = item.get("content")

      if role == "user":
          # Extract all text parts of user content
          user_texts = []
          for part in content:
              if part.get("type") == "text":
                  user_texts.append(part.get("text", ""))
          user_prompt = "\n".join(user_texts).strip()

      elif role == "assistant":
          # Assistant content is often plain text
          if isinstance(content, str):
              assistant_response = content.strip()
          elif isinstance(content, list):
              texts = [c.get("text", "") for c in content if c.get("type") == "text"]
              assistant_response = "\n".join(texts).strip()
          else:
              assistant_response = str(content)

  return {"user_prompt": user_prompt, "output": assistant_response, "file": filename}
