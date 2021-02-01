import math
import os
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import datetime
from matplotlib.backends.backend_pdf import PdfPages

PDF_PATH = './res/statistical_report.pdf'

class PDF():
  def __init__(self, clientId, printPdf=False):
    self.clientId = clientId
    self.printPdf = printPdf

    if self.printPdf is True:
      self.pdf = PdfPages(PDF_PATH)
      self.saveFigInPdf(self.generatePdfHomepage())
    else:
      self.removePdf()

  def closePdf(self):
    """Close the pdf"""
    self.pdf.close()

  def removePdf(self):
    """Delete the pdf"""
    os.remove(PDF_PATH)

  def saveFigPng(self, fig, name):
    """Save a fig into png"""
    fig.savefig(f'./res/{name}.png')

  def saveFigInPdf(self, fig):
    """Save a fig in pdf"""
    if self.printPdf is True:
      self.pdf.savefig(fig)

  def generatePdfHomepage(self):
    fig = plt.figure()
    text = fig.text(0.5, 0.5, f'T-DAT-901\nRapport statistique\nClient #{self.clientId}\n\n'
    'Produit par Maxime Gavens, Théo Walcker,\n Jean Bosc, Arnaud Brown et Mathieu Dufour\n\n'
    f'Le {datetime.date.today()}',
                  ha='center', va='center', size=20)
    # text.set_path_effects([path_effects.Normal()])
    return fig

  def setPdfMetadata(self):
    """Set the pdf's metadata"""
    d = self.pdf.infodict()
    d['Title'] = 'Statistical Report'
    d['Author'] = u'Théo Walcker, Maxime Gavens, Arnaud Brown, Jean Bosc & Mathieu Dufour'
    d['Subject'] = f'A statistical report of the client'
    d['CreationDate'] = datetime.datetime.today()