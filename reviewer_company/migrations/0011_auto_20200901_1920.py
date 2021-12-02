# Generated by Django 3.1 on 2020-09-01 13:50

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('reviewer', '0010_review_done'),
    ]

    operations = [
        migrations.CreateModel(
            name='Image_upload',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=200)),
                ('description', models.CharField(max_length=2000)),
                ('image1', models.ImageField(upload_to='images')),
                ('image2', models.ImageField(upload_to='images')),
                ('image3', models.ImageField(upload_to='images')),
                ('company_id', models.CharField(max_length=10)),
            ],
        ),
        migrations.DeleteModel(
            name='Image1',
        ),
    ]