# Generated by Django 4.0.2 on 2022-02-12 06:07

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='assignment',
            name='has_answer',
            field=models.BooleanField(default=False),
        ),
    ]