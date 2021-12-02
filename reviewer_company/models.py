from django.db import models

# Create your models here.
class review(models.Model):
    name=models.CharField(max_length=100)
    address=models.CharField(max_length=100)

class record_reaction(models.Model):
    reaction=models.CharField(max_length=100)
    product_id=models.CharField(max_length=20)
    reviewer_id=models.CharField(max_length=20)
class use(models.Model):
    email=models.CharField(max_length=100)
    password=models.CharField(max_length=20)
class Image_upload(models.Model):
    title = models.CharField(max_length=200)
    description=models.CharField(max_length=2000)
    image1 = models.ImageField(upload_to='images')
    image2 = models.ImageField(upload_to='images')
    image3 = models.ImageField(upload_to='images')

    company_id=models.CharField(max_length=10)
    product_type=models.CharField(max_length=100,default="sports")
    question1=models.CharField(max_length=300,default="")
    question2=models.CharField(max_length=300,default="")
    question3=models.CharField(max_length=300,default="")
    question4=models.CharField(max_length=300,default="")
    question5=models.CharField(max_length=300,default="")
    question6=models.CharField(max_length=300,default="")
    question7=models.CharField(max_length=300,default="")
    question8=models.CharField(max_length=300,default="")

class reviewer_info(models.Model):
    name=models.CharField(max_length=100)
    email=models.EmailField()
    password=models.CharField(max_length=20)
    phone=models.BigIntegerField()
    date=models.DateField(auto_now_add= True )
    product_type=models.CharField(max_length=100,default="food")
class company_info(models.Model):
    name=models.CharField(max_length=100)
    email=models.EmailField()
    password=models.CharField(max_length=20)
    phone=models.BigIntegerField()
    date=models.DateField(auto_now_add= True )
class question_answer(models.Model):
    product_id=models.CharField(max_length=20)
    question1=models.CharField(max_length=20)
    question2=models.CharField(max_length=20)
    question3=models.CharField(max_length=20)
    question4=models.CharField(max_length=2000,default="")
    question5=models.CharField(max_length=2000,default="")
    question6=models.CharField(max_length=2000,default="")
    question7=models.CharField(max_length=2000,default="")
    question8=models.CharField(max_length=2000,default="")
    question9=models.CharField(max_length=2000,default="")
    question10=models.CharField(max_length=2000,default="")
class review_done(models.Model):
    product_id=models.CharField(max_length=20)
    reviewer_id=models.CharField(max_length=20)

class review_incentive(models.Model):
    product_id=models.CharField(max_length=20)
    reviewer_id=models.CharField(max_length=20)
    product_name=models.CharField(max_length=200)
    company_name=models.CharField(max_length=200)
    coupon_code=models.CharField(max_length=20)
class record_sentiment(models.Model):
    sentiment=models.CharField(max_length=100)
    product_id=models.CharField(max_length=20)
    reviewer_id=models.CharField(max_length=20)
