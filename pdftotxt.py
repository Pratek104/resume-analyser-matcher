from PyPDF2 import PdfReader
reader = PdfReader('resume.pdf')
full_text = ""
for page in reader.pages:
    text = page.extract_text()
    if text: 
        full_text += text
words = full_text.split()
limited_text = ' '.join(words[:10000])
print(limited_text)


