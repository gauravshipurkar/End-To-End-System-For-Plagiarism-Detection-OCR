# Generated by Django 4.0.2 on 2022-03-15 10:24

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0004_remove_page_path'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='assignment',
            name='name',
        ),
    ]
