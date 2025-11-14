from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
import io

def generate_postanovlenie_pdf(content: str) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    text_object = c.beginText(25 * mm, height - 25 * mm)
    text_object.setFont("Times-Roman", 12)

    for line in content.split("\n"):
        text_object.textLine(line)

    c.drawText(text_object)
    c.showPage()
    c.save()
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data
