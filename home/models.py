from django.db import models
from django.conf import settings
import os
# Create your models here.



class Assignment(models.Model):
    answer_key = models.TextField()
    has_answer = models.BooleanField(default=False)

    def __str__(self):
        return f'{self.pk}'

class Pdf(models.Model):
    name = models.CharField(max_length=256, blank=False)
    file = models.FileField(upload_to='pdfs') # TODO: specify path
    assignment = models.ForeignKey(Assignment, on_delete=models.CASCADE)
    
    # storing the recognize parts
    text = models.TextField()
    spell_corrected = models.TextField()

    def __str__(self):
        return f'{self.name}'

class Page(models.Model):
    # storing the recognize parts
    text = models.TextField()
    spell_corrected = models.TextField()
    #path = models.CharField(max_length=256, blank=False)
    #image =  models.FileField() # TODO: specify path
    page_num = models.IntegerField(blank=True, default=-1)
    pdf = models.ForeignKey(Pdf, on_delete=models.CASCADE)

    @property
    def getpath(self):
        base = settings.MEDIA_ROOT
        imagefolder = os.path.join(base, 'images')
        pdfloc = os.path.join(imagefolder, str(self.pdf.pk))
        imagepath = os.path.join(pdfloc, str(self.page_num) + '.png')
        return imagepath
    
    def __str__(self):
        return f'{self.page_num}'

class Word(models.Model):
    order =  models.IntegerField(blank=True)
    page =  models.ForeignKey(Page, on_delete=models.CASCADE)
    text = models.CharField(max_length=256, blank=False)
    
