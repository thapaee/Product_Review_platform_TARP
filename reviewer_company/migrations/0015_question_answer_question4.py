# Generated by Django 3.1 on 2020-09-03 12:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('reviewer', '0014_auto_20200902_2143'),
    ]

    operations = [
        migrations.AddField(
            model_name='question_answer',
            name='question4',
            field=models.CharField(default='', max_length=2000),
        ),
    ]
