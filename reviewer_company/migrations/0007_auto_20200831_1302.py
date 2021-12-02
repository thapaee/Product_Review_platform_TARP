# Generated by Django 3.1 on 2020-08-31 07:32

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('reviewer', '0006_company_info'),
    ]

    operations = [
        migrations.CreateModel(
            name='Image1',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=200)),
                ('image1', models.ImageField(upload_to='images')),
                ('image2', models.ImageField(upload_to='images')),
                ('company_id', models.CharField(max_length=10)),
            ],
        ),
        migrations.DeleteModel(
            name='Image',
        ),
    ]
