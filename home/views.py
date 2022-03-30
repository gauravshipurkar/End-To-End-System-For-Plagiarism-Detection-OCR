from importlib.metadata import requires
from unittest import result
from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
from matplotlib.style import context
from home.models import *
import numpy as np
# Create your views here.

# utils
import glob, sys, fitz
import os

from .utlis import *


def home(request):
     #return HttpResponse("Hello, world. You're at the polls index.")
     return render(request, 'index.html')

def results(request):
     context = {
          'final_list': [['download (3)-converted.pdf', 'download (2)-converted.pdf', 0.32636649538760315], ['download (5)-converted.pdf', 'download (2)-converted.pdf', 0.13529079526307547], ['download (5)-converted.pdf', 'download (3)-converted.pdf', 0.09531714233741803]],
          'haskey': True,
          'answer_list': [['download (2)-converted.pdf', 0.22724574274425494], ['download (3)-converted.pdf', 0.1427260491035693], ['download (5)-converted.pdf', 0.0]]
     }

     return render(request, 'results.html', context)

def check(request):
     print('hererehre')
     if request.method == 'POST':
          files = request.FILES.getlist('files')
          # assignmentname = request.POST['assignment']
          answerkey = request.POST['answerkey']
          has_answer = True
          if answerkey == '':
               has_answer = False
     
          print(files)


          ########################################################
          # TODO: error check
          ########################################################




          

          ########################################################
          # saving the assignment
          newass = Assignment(answer_key=answerkey, has_answer=has_answer)
          newass.save()
          ########################################################
          




          ########################################################
          # saving pdfs and images
          # making the image folder if not already there:
           
     
          try: 
               storing = settings.MEDIA_ROOT
               rootimagefolder = os.path.join(storing, str('images'))
               os.mkdir(rootimagefolder)
          except:
               print('folder already there, no need to make a new one')



          # this is to break the pdf into images
          # saving the files
          mat = fitz.Matrix(2, 2) # random shit
          for file in files:
               rootimagefolder = os.path.join(storing, str('images'))
               # making a pdf object and storing the file it
               pdfs = Pdf(name=str(file), file=file, assignment = newass)
               pdfs.save()

               pdf_path = pdfs.file.path # path to access the pdf file
               images = fitz.open(pdf_path)

               # making a separate folder for this pdf file in the images directory 
               rootimagefolder = os.path.join(rootimagefolder, str(pdfs.pk))
               os.mkdir(rootimagefolder)
               for page in images:
                    pix = page.get_pixmap(matrix=mat)  # render page to an image                    
                    pp = os.path.join(rootimagefolder, str(str(page.number) + '.png'))
                    pix.save(pp)  # store image as a PNG
                    newimage = Page(page_num=page.number, pdf=pdfs)
                    newimage.save()

                    #print(page2word(newimage.getpath))

          ########################################################


          # now the images are saved we need to use page to word algorithm here

          '''
          algorithm
          functions list:
          page2word
          wordrecog
          speller
          plagia

          for each pdf in assignment
               tea = []
               coffee = []
               for each page in pdf
                    words_page = page2word(page) # histogram
                    recognized_words = wordrecog(words_page) # cnn + lstm

                    
                    page.text = recognized_words.str
                    page.spell_corrected = speller(page.text)
                    page.save()
               
                    tea.append(page.text)
                    coffee.append(page.spell_corrected)

               pdf.text = tea.str
               pdf.spell_corrected = coffee.str
               pdf.save()

               # so far all the things are converted to spell corrected text

          pdfs = assignment.pdfs
          output = plagia(pdfs)
          numppee = np.zeros((len(pdfs),len(pdfs)))

          for i, pdf in pdfs.enumerate:
               for j, pdf2 in pdfs.enumerate:
                    numppee[i,j] = plagia(pdf, pdf2)



          output:

          

          '''


          #print(newass.pdf_set.all())

          pdfs = newass.pdf_set.all()
          documents = []
          filename = []
          for pdf in pdfs:
               tea = []
               coffee = []
               pages = pdf.page_set.all()
               for page in pages:
                    print(page.getpath)
                    fwi = page2word(page.getpath)
                    text = wordrecog(fwi)
                    ct = speller(text)

                    page.text = text
                    page.spell_corrected = ct
                    page.save()

                    tea.append(page.text)
                    coffee.append(ct)

               pdf.text = ''.join(tea)
               pdf.spell_corrected = ''.join(coffee)


               





                    # now all we need to do is put this image into page to word

               pdf.save()
               print(pdf.text)
               print(pdf.spell_corrected)
               documents.append(pdf.spell_corrected)
               filename.append(pdf.name)
          


          plags = process_tfidf_similarity(documents)
          print(plags)
          print(filename)

          answerlist = None
          newal = []

          if newass.has_answer:
               answerlist = tfidf_answerkey(documents, newass.answer_key)
               # compare it with answer as well

               for a in range(len(answerlist)):
                    newal.append([filename[a], answerlist[a]])

          npp = plags


          docs = filename
          ld = len(docs)
          npp = npp.reshape(ld,ld)

          final_list = []

          for i in range(ld):
               for j in range(ld):
                    if i >= j:
                         continue
                    
                    final_list.append([docs[i],docs[j],npp[i,j]])


          # print(final_list)

          litt = sorted(final_list, key = lambda x: x[2], reverse=True)
          answerlist = sorted(newal, key = lambda x: x[1], reverse=True)
          # print(litt)



          context = {
               'final_list': litt,
               'haskey': newass.has_answer,
               'answer_list': answerlist
          }

          return render(request, 'results.html', context)

               
          
          





          ########################################################

          

               
          return HttpResponse(plags)
     else:
          return HttpResponse('Error not a post')