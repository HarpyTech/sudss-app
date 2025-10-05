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
