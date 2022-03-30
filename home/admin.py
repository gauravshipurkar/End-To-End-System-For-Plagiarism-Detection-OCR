from django.contrib import admin

# Register your models here.
from .models import *

# Register your models here.
admin.site.register(Assignment)
admin.site.register(Pdf)
admin.site.register(Page)
admin.site.register(Word)
