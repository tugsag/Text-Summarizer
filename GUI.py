import tkinter as tk
from Image import read_image
import Keyphrase
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage


m = tk.Tk()
m.title('Summary')
m.geometry('1500x500')
string = ''
filechosen = False

def choose_file():
    global string
    global filechosen
    filename = tk.filedialog.askopenfilename()
    if filename=='':
        filedisp['text'] = "Please choose an image file or a text file (pdf or txt)."

    elif '.pdf' in filename:
        filechosen = True
        filedisp['text'] = "Chosen file: " + filename
        manager = PDFResourceManager()
        retstr = BytesIO()
        layout = LAParams(all_texts=True)
        device = TextConverter(manager, retstr, laparams=layout)
        filepath = open(filename, 'rb')
        interpreter = PDFPageInterpreter(manager, device)

        for page in PDFPage.get_pages(filepath, check_extractable=True):
            interpreter.process_page(page)

        text = retstr.getvalue()

        filepath.close()
        device.close()
        retstr.close()

        try:
            text = text.decode()
        except AttributeError:
            pass

        string = text

    elif '.txt' in filename:
        filechosen = True
        filedisp['text'] = "Chosen file: " + filename
        with open(filename, 'r') as f:
            f1 = f.read()
            string = f1

    elif '.jpg' in filename or '.png' in filename:
        filechosen = True
        filedisp['text'] = "Chosen file: " + filename
        string = read_image(filename)
    else:
        filedisp['text'] = 'Please choose a valid file'


def keyword_extraction():
    if filechosen == False:
        filedisp['text'] = 'Please choose a valid file'
    else:
        top = Keyphrase.score_keyphrases(string)
        keywords = ', '.join(top)
        kedisp['text'] = keywords

def summarize():
    if not filechosen:
        filedisp['text'] = 'Please choose a valid file'
    else:
        summary_list = Keyphrase.generate_summary(string, 5)
        summary = '\n'.join(summary_list)
        sdisp['text'] = summary
        
        

## Buttons
##-------------
    
choose_file_button = tk.Button(m, text='Choose File', width=25, command=choose_file)
choose_file_button.grid(row = 0, column = 0)

keyword_extraction_button = tk.Button(m, text='Keyword Extract', width = 25, command=keyword_extraction)
keyword_extraction_button.grid(row = 1, column = 0)

summarize_button = tk.Button(m, text='Summarize', width = 25, command=summarize)
summarize_button.grid(row = 2, column = 0)

## Displays
##-------------

filedisp = tk.Label(m)
filedisp.grid(row = 0, column = 1)

kedisp = tk.Label(m)
kedisp.grid(row=1, column=1)

sdisp = tk.Label(m)
sdisp.grid(row=2, column=1)

m.mainloop()
