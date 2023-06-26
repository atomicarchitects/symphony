from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image
from reportlab.lib import colors


def create_pdf_with_table_and_images(output_file):
    doc = SimpleDocTemplate(output_file, pagesize=letter)
    elements = []

    # Adding text
    text_content = [
        ["Name", "Age", "Country"],
        ["John", "25", "USA"],
        ["Alice", "30", "Canada"],
        ["Bob", "35", "UK"],
    ]

    # Create a table style
    table_style = TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.white),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.gray),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 14),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
            ("BACKGROUND", (0, 1), (-1, -1), colors.white),
            ("TEXTCOLOR", (0, 1), (-1, -1), colors.black),
            ("ALIGN", (0, 1), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 1), (-1, -1), 12),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
        ]
    )

    # Create the table and apply the style
    table = Table(text_content)
    table.setStyle(table_style)
    elements.append(table)

    # Adding images
    image_path = "/Users/ameyad/Documents/spherical-harmonic-net/analyses/outputs/v2/num_params/e3schnet_test_atom_type_loss_params.png"
    image = Image(image_path)
    image.drawWidth = 3 * inch  # Adjust the width of the image as needed
    image.drawHeight = 3 * inch  # Adjust the height of the image as needed
    elements.append(image)

    # Build the PDF document
    doc.build(elements)


# Usage
output_file = "output.pdf"
create_pdf_with_table_and_images(output_file)
