# Generated by Django 3.1 on 2020-09-01 03:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('reviewer', '0008_auto_20200831_2326'),
    ]

    operations = [
        migrations.CreateModel(
            name='questions_answer',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('product_id', models.CharField(max_length=20)),
                ('question1', models.CharField(max_length=20)),
                ('question2', models.CharField(max_length=20)),
            ],
        ),
    ]
