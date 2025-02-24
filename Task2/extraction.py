import fitz  # PyMuPDF for text extraction
import pdfplumber  # For table extraction
import json
import os

def extract_text(pdf_path):
    extracted_text = []
    
    try:
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text").strip()
                if text:
                    extracted_text.append(f"\nPage {page_num}\n{text}")
        
        return "\n".join(extracted_text).strip()
    
    except Exception as e:
        print(f" Error extracting text: {e}")
        return ""

def extract_tables(pdf_path):
    tables = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                table = page.extract_table()
                if table:
                    tables.append(table)
        return tables
    
    except Exception as e:
        print(f" Error extracting tables: {e}")
        return []

def save_to_json(text, tables, output_file="document_extraction_JSON.json"):
    data = {
        "Headers": text,
        "List_items": tables
    }
    
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

    print(f"\n JSON file created {output_file}")

if __name__ == "__main__":
    pdf_path = input(" Enter the full path of the PDF file: ").strip()

    if not os.path.isfile(pdf_path):
        print(" Please enter a correct file path.")
    else:

        extracted_text = extract_text(pdf_path)
        extracted_tables = extract_tables(pdf_path)
        if extracted_text:
            print(" Extracted Text from PDF:\n")
            print(extracted_text)
        else:
            print(" No text found in the PDF.")

        if extracted_tables:
            print("\n Extracted Tables from PDF:")
            for idx, table in enumerate(extracted_tables, start=1):
                print(f"\n--- Table {idx} ---")
                for row in table:
                    print(row)
        else:
            print("\n No tables found in the PDF.")

        save_to_json(extracted_text, extracted_tables)

        print("\n successfully extracted text and tables from the PDF file.")