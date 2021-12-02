from django.shortcuts import render
from .models import record_sentiment,review_incentive,record_reaction,use,Image_upload,reviewer_info,company_info,question_answer,review_done,review_incentive
import random
from keras.models import load_model
import cv2
import numpy as np
from keras.preprocessing import image
from textblob import TextBlob
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Create your views here.
def capp(request):
    #from fer import FER
    emotion_count=[0,0,0,0,0,0,0]
    a=0

    model = load_model('weights_min_loss1.hdf5')
    face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    video_capture = cv2.VideoCapture(0)
    while True:

        ret, test_img = video_capture.read()
        a+=1
        if(a==250):
           break
        gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)



        for (x,y,w,h) in faces_detected:
            cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
            roi_gray=gray_img[y:y+w,x:x+h]
            roi_gray=cv2.resize(roi_gray,(48,48))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis = 0)
            img_pixels /= 255

            predictions = model.predict(img_pixels)
            max_index = np.argmax(predictions[0])
            predicted_emotion = emotions[max_index]
            if(predicted_emotion=='happy'):
                  emotion_count[3]+=1
            elif(predicted_emotion=='neutral'):
                  emotion_count[6]+=1
            elif(predicted_emotion=='angry'):
                  emotion_count[0]+=1
            elif(predicted_emotion=='surprise'):
                  emotion_count[6]+=1
            cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)


        resized_img = cv2.resize(test_img, (1000, 700))
        cv2.imshow('Facial emotion analysis ',resized_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cv2.destroyAllWindows()

    p= record_reaction.objects.create(reaction=emotions[emotion_count.index(max(emotion_count))],product_id=request.POST['pid'],reviewer_id=request.session['user'])
    return render(request,'after_record.html',{'pid':request.POST['pid']})


def cap(request):
    #from fer import FER
    emotion_count=[0,0,0,0,0,0,0]
    a=0

    model = load_model(os.path.join(BASE_DIR,'static/weights_min_loss1.hdf5'))
    face_haar_cascade = cv2.CascadeClassifier(os.path.join(BASE_DIR,'static/haarcascade_frontalface_default.xml'))

    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    video_capture = cv2.VideoCapture(0)
    while True:

        ret, test_img = video_capture.read()
        a+=1
        if(a==250):
           break
        gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)



        for (x,y,w,h) in faces_detected:
            cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
            roi_gray=gray_img[y:y+w,x:x+h]
            roi_gray=cv2.resize(roi_gray,(48,48))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis = 0)
            img_pixels /= 255

            predictions = model.predict(img_pixels)
            max_index = np.argmax(predictions[0])
            predicted_emotion = emotions[max_index]
            if(predicted_emotion=='happy'):
                  emotion_count[3]+=1
            elif(predicted_emotion=='neutral'):
                  emotion_count[6]+=1
            elif(predicted_emotion=='angry'):
                  emotion_count[0]+=1
            elif(predicted_emotion=='surprise'):
                  emotion_count[6]+=1
            cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)


        resized_img = cv2.resize(test_img, (1000, 700))
        cv2.imshow('Facial emotion analysis ',resized_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cv2.destroyAllWindows()
    p=Image_upload.objects.filter(id=request.POST['pid'])
    #p= record_reaction.objects.create(reaction=emotions[emotion_count.index(max(emotion_count))],product_id=request.POST['pid'],reviewer_id=request.session['user'])
    return render(request,'read_description.html',{'p':p[0],'pid':request.POST['pid'],'ptype':request.POST['ptype'],'reaction':emotions[emotion_count.index(max(emotion_count))]})

def read_description(request):
        return render(request,'after_record.html',{'pid':request.POST['pid'],'ptype':request.POST['ptype'],'reaction':request.POST['reaction']})



def review(request):
    return render(request,'reviewer_dash.html')
def reviewer_signup(request):
    if request.method=="POST":
        name=request.POST['name']
        email=request.POST['email']
        pwd=request.POST['password']
        phone=request.POST['phone']
        ty=request.POST.getlist('sports')
        sp=""
        for x in ty:
            sp+=x
            sp+=","
        sp=sp[0:len(sp)-1]
        print(sp)
        p=reviewer_info.objects.create(name=name,email=email,password=pwd,phone=phone,product_type=sp)
        request.session["user"]=p.id
        request.session["name"]=request.POST['name']
        request.session["typ"]=sp.split(",")
        ii=review_incentive.objects.filter(reviewer_id=request.session['user'])
        oo=review_done.objects.filter(reviewer_id=request.session['user'])
        z=[]
        for i in oo:
            z.append(i.product_id)
        o=Image_upload.objects.exclude(id__in=z)
        k=[]
        for x in o:
            splitname=(x.product_type).split(",")
            for xx in splitname:
                if xx in request.session['typ']:
                    k.append(x)
                    break
        print(k)
        return render(request,'reviewer_dash.html',{'review_no':len(k),'incentive_no':len(ii)})
        return render(request,'reviewer_dash.html',{'review_no':0,'incentive_no':0})
    return render(request,'signup.html')
def company_signup(request):
    if request.method=="POST":
        name=request.POST['name']
        email=request.POST['email']
        pwd=request.POST['password']
        phone=request.POST['phone']

        p=company_info.objects.create(name=name,email=email,password=pwd,phone=phone)
        request.session["user"]=p.id
        request.session["name"]=request.POST['name']
        return render(request,'company_dash.html',{'total_review':0,'total_product':0})
    return render(request,'company_signup.html')
def add_to_model(request):
    if(request.method=='POST'):
         p=use.objects.create(email=request.POST['email'],password=request.POST['pwd'])
    return render(request,'reviewer_dash.html')
def company_dash(request):

    return render(request,'company_dash.html')
def add_product(request):
    return render(request,'company_upload.html')

def add_final(request):
    message=""


    return render(request,'addd.html',{'x':request.POST.get('sports'),'title':request.POST['a'],'description':request.POST['des']})

def sucess(request):
    p=Image_upload.objects.create(title=request.POST['title'],description=request.POST['des'],product_type=request.POST['type'],company_id=request.session['user'],image1=request.FILES['image1'],image2=request.FILES['image2'],image3=request.FILES['image3'],question1=request.POST['1'],question2=request.POST['2'],question3=request.POST['3'],question4=request.POST['4'],question5=request.POST['5'],question6=request.POST['6'],question7=request.POST['7'],question8=request.POST['8'])

    return render(request,'s.html')
def add_fina(request):
    message=""
    if request.method=="POST":
        ty=request.POST.get('sports')
        sp=ty
        p=Image_upload.objects.create(title=request.POST['a'],description=request.POST['des'],product_type=sp,one=request.POST['1'],two=request.POST['2'],three=request.POST['3'],four=request.POST['4'],five=request.POST['5'],six=request.POST['6'],seven=request.POST['7'],eight=request.POST['8'])
        a=1
        message="upload Sucessful"
    return render(request,'company_upload.html',{'message':message})
def choose(request):
    return render(request,'landing.html')
def view_product(request):
    b=Image_upload.objects.all()
    c=review_done.objects.filter(reviewer_id=request.session['user'])
    z=[]
    e=[]
    for u in c:
        k=int(u.product_id)
        z.append(k)

    for x in b:
        if x.id not in z:
            splitset=(x.product_type).split(",")
            v=0
            for xx in splitset:
                if xx in request.session['typ']:
                    v=1
                    break;
            if v==1:
                e.append(x)
    if len(e)==0:
        f=True
    else:
        f=False
    return render(request,'view_product.html',{'b':e,'f':f})

def reviewer_login(request):
    message=""
    if(request.method=='POST'):
        p=reviewer_info.objects.filter(email=request.POST['email'],password=request.POST['pwd'])

        if(len(p)==1):
            request.session['user']=p[0].id
            request.session['name']=p[0].name
            request.session["typ"]=(p[0].product_type).split(",")
            ii=review_incentive.objects.filter(reviewer_id=request.session['user'])
            oo=review_done.objects.filter(reviewer_id=request.session['user'])
            z=[]
            for i in oo:
                z.append(i.product_id)
            o=Image_upload.objects.exclude(id__in=z)
            k=[]
            for x in o:
                splitname=(x.product_type).split(",")
                for xx in splitname:
                    if xx in request.session['typ']:
                        k.append(x)
                        break
            print(k)
            return render(request,'reviewer_dash.html',{'review_no':len(k),'incentive_no':len(ii)})
        message="Invalid email or password"
    return render(request,'login.html',{'message':message})

def company_login(request):
    message=""
    if(request.method=='POST'):
        p=company_info.objects.filter(email=request.POST['email'],password=request.POST['pwd'])

        if(len(p)==1):
            request.session['user']=p[0].id
            request.session['name']=p[0].name
            o=Image_upload.objects.filter(company_id=p[0].id)
            u=[]
            for x in o:
                u.append(x.id)
            q=question_answer.objects.filter(product_id__in=u)

            return render(request,'company_dash.html',{'total_review':len(q),'total_product':len(o)})
        message="Invalid email or password"

    return render(request,'company_login.html',{'message':message})

def signup_landing(request):
    return render(request,'landing_signup.html')


def record_expression(request):
    p=Image_upload.objects.filter(id=request.POST["pid"])
    pid=request.POST["pid"]
    ptype=request.POST["ptype"]
    return render(request,'record_expression.html',{'p':p[0],'pid':pid,'ptype':ptype, 'pname':request.POST["pname"]})
def company_visualize(request):

    p=record_reaction.objects.filter(reaction="sad",product_id=request.POST['pid'])
    sad=len(p)
    p=record_reaction.objects.filter(reaction="angry",product_id=request.POST['pid'])
    angry=len(p)
    p=record_reaction.objects.filter(reaction="happy",product_id=request.POST['pid'])
    happy=len(p)
    p=record_reaction.objects.filter(reaction="neutral",product_id=request.POST['pid'])
    neutral=len(p)
    p=record_reaction.objects.filter(reaction="suprise",product_id=request.POST['pid'])
    suprise=len(p)
    if(request.POST["ptype"]=="food"):
        yes1=question_answer.objects.filter(product_id=request.POST["pid"],question1="yes")
        no1=question_answer.objects.filter(product_id=request.POST["pid"],question1="no")
        yes1=len(yes1)
        no1=len(no1)
        yes2=question_answer.objects.filter(product_id=request.POST["pid"],question2="yes")
        no2=question_answer.objects.filter(product_id=request.POST["pid"],question2="no")
        yes2=len(yes2)
        no2=len(no2)
        yes3=question_answer.objects.filter(product_id=request.POST["pid"],question3="yes")
        no3=question_answer.objects.filter(product_id=request.POST["pid"],question3="no")
        yes3=len(yes3)
        no3=len(no3)
        yes5=question_answer.objects.filter(product_id=request.POST["pid"],question5="yes")
        no5=question_answer.objects.filter(product_id=request.POST["pid"],question5="no")
        maybe5=question_answer.objects.filter(product_id=request.POST["pid"],question5="maybe")
        yes5=len(yes5)
        no5=len(no5)
        maybe5=len(maybe5)

        yes8=question_answer.objects.filter(product_id=request.POST["pid"],question8="yes")
        no8=question_answer.objects.filter(product_id=request.POST["pid"],question8="no")
        yes8=len(yes8)
        no8=len(no8)
        excellent=question_answer.objects.filter(product_id=request.POST["pid"],question10="Excellent")
        good=question_answer.objects.filter(product_id=request.POST["pid"],question10="Good")
        satisfactory=question_answer.objects.filter(product_id=request.POST["pid"],question10="Satisfactory")
        poor=question_answer.objects.filter(product_id=request.POST["pid"],question10="Poor")
        verypoor=question_answer.objects.filter(product_id=request.POST["pid"],question10="Verypoor")
        excellent=len(excellent)
        good=len(good)
        satisfactory=len(satisfactory)
        poor=len(poor)
        verypoor=len(verypoor)
        positive=record_sentiment.objects.filter(product_id=request.POST["pid"],sentiment="positive")
        negative=record_sentiment.objects.filter(product_id=request.POST["pid"],sentiment="negative")
        neutral1=record_sentiment.objects.filter(product_id=request.POST["pid"],sentiment="neutral")
        negative=len(negative)
        positive=len(positive)
        neutral1=len(neutral1)
    elif(request.POST["ptype"]=="vechile"):
        yes1=question_answer.objects.filter(product_id=request.POST["pid"],question1="yes")
        no1=question_answer.objects.filter(product_id=request.POST["pid"],question1="no")
        yes1=len(yes1)
        no1=len(no1)
        yes2=question_answer.objects.filter(product_id=request.POST["pid"],question2="yes")
        no2=question_answer.objects.filter(product_id=request.POST["pid"],question2="no")
        yes2=len(yes2)
        no2=len(no2)
        yes3=question_answer.objects.filter(product_id=request.POST["pid"],question3="yes")
        no3=question_answer.objects.filter(product_id=request.POST["pid"],question3="no")
        yes3=len(yes3)
        no3=len(no3)
        yes4=question_answer.objects.filter(product_id=request.POST["pid"],question4="yes")
        no4=question_answer.objects.filter(product_id=request.POST["pid"],question4="no")
        yes4=len(yes4)
        no4=len(no4)
        yes5=question_answer.objects.filter(product_id=request.POST["pid"],question5="yes")
        no5=question_answer.objects.filter(product_id=request.POST["pid"],question5="no")
        maybe5=question_answer.objects.filter(product_id=request.POST["pid"],question5="maybe")
        yes5=len(yes5)
        no5=len(no5)
        maybe5=len(maybe5)
        yes7=question_answer.objects.filter(product_id=request.POST["pid"],question7="yes")
        no7=question_answer.objects.filter(product_id=request.POST["pid"],question7="no")
        maybe7=question_answer.objects.filter(product_id=request.POST["pid"],question7="maybe")
        yes7=len(yes7)
        no7=len(no7)
        maybe7=len(maybe7)
        yes9=question_answer.objects.filter(product_id=request.POST["pid"],question9="yes")
        no9=question_answer.objects.filter(product_id=request.POST["pid"],question9="no")
        yes9=len(yes9)
        no9=len(no9)
        excellent=question_answer.objects.filter(product_id=request.POST["pid"],question10="Excellent")
        good=question_answer.objects.filter(product_id=request.POST["pid"],question10="Good")
        satisfactory=question_answer.objects.filter(product_id=request.POST["pid"],question10="Satisfactory")
        poor=question_answer.objects.filter(product_id=request.POST["pid"],question10="Poor")
        verypoor=question_answer.objects.filter(product_id=request.POST["pid"],question10="Verypoor")
        excellent=len(excellent)
        good=len(good)
        satisfactory=len(satisfactory)
        poor=len(poor)
        verypoor=len(verypoor)
        positive=record_sentiment.objects.filter(product_id=request.POST["pid"],sentiment="positive")
        negative=record_sentiment.objects.filter(product_id=request.POST["pid"],sentiment="negative")
        neutral1=record_sentiment.objects.filter(product_id=request.POST["pid"],sentiment="neutral")
        negative=len(negative)
        positive=len(positive)
        neutral1=len(neutral1)
    elif(request.POST["ptype"]=="clothes"):
        yes1=question_answer.objects.filter(product_id=request.POST["pid"],question1="yes")
        no1=question_answer.objects.filter(product_id=request.POST["pid"],question1="no")
        yes1=len(yes1)
        no1=len(no1)
        yes2=question_answer.objects.filter(product_id=request.POST["pid"],question2="yes")
        no2=question_answer.objects.filter(product_id=request.POST["pid"],question2="no")
        yes2=len(yes2)
        no2=len(no2)
        yes3=question_answer.objects.filter(product_id=request.POST["pid"],question3="yes")
        no3=question_answer.objects.filter(product_id=request.POST["pid"],question3="no")
        yes3=len(yes3)
        no3=len(no3)
        yes4=question_answer.objects.filter(product_id=request.POST["pid"],question4="yes")
        no4=question_answer.objects.filter(product_id=request.POST["pid"],question4="no")
        yes4=len(yes4)
        no4=len(no4)
        yes7=question_answer.objects.filter(product_id=request.POST["pid"],question7="yes")
        no7=question_answer.objects.filter(product_id=request.POST["pid"],question7="no")
        yes7=len(yes7)
        no7=len(no7)
        summer=question_answer.objects.filter(product_id=request.POST["pid"],question8="summer")
        winter=question_answer.objects.filter(product_id=request.POST["pid"],question8="winter")
        summer=len(summer)
        winter=len(winter)
        excellent=question_answer.objects.filter(product_id=request.POST["pid"],question10="Excellent")
        good=question_answer.objects.filter(product_id=request.POST["pid"],question10="Good")
        satisfactory=question_answer.objects.filter(product_id=request.POST["pid"],question10="Satisfactory")
        poor=question_answer.objects.filter(product_id=request.POST["pid"],question10="Poor")
        verypoor=question_answer.objects.filter(product_id=request.POST["pid"],question10="Verypoor")
        excellent=len(excellent)
        good=len(good)
        satisfactory=len(satisfactory)
        poor=len(poor)
        verypoor=len(verypoor)
        positive=record_sentiment.objects.filter(product_id=request.POST["pid"],sentiment="positive")
        negative=record_sentiment.objects.filter(product_id=request.POST["pid"],sentiment="negative")
        neutral1=record_sentiment.objects.filter(product_id=request.POST["pid"],sentiment="neutral")
        negative=len(negative)
        positive=len(positive)
        neutral1=len(neutral1)
    elif(request.POST["ptype"]=="tech"):
        yes1=question_answer.objects.filter(product_id=request.POST["pid"],question1="yes")
        no1=question_answer.objects.filter(product_id=request.POST["pid"],question1="no")
        yes1=len(yes1)
        no1=len(no1)
        yes2=question_answer.objects.filter(product_id=request.POST["pid"],question2="yes")
        no2=question_answer.objects.filter(product_id=request.POST["pid"],question2="no")
        yes2=len(yes2)
        no2=len(no2)
        design=question_answer.objects.filter(product_id=request.POST["pid"],question3="design")
        features=question_answer.objects.filter(product_id=request.POST["pid"],question3="features")
        both=question_answer.objects.filter(product_id=request.POST["pid"],question3="both")
        design=len(design)
        features=len(features)
        both=len(both)
        yes4=question_answer.objects.filter(product_id=request.POST["pid"],question4="yes")
        no4=question_answer.objects.filter(product_id=request.POST["pid"],question4="no")
        yes4=len(yes4)
        no4=len(no4)
        one=question_answer.objects.filter(product_id=request.POST["pid"],question5="1")
        two=question_answer.objects.filter(product_id=request.POST["pid"],question5="2")
        three=question_answer.objects.filter(product_id=request.POST["pid"],question5="3")
        four=question_answer.objects.filter(product_id=request.POST["pid"],question5="4")
        five=question_answer.objects.filter(product_id=request.POST["pid"],question5="5")
        one=len(one)
        two=len(two)
        three=len(three)
        four=len(four)
        five=len(five)
        yes7=question_answer.objects.filter(product_id=request.POST["pid"],question7="yes")
        no7=question_answer.objects.filter(product_id=request.POST["pid"],question7="no")
        yes7=len(yes7)
        no7=len(no7)
        excellent=question_answer.objects.filter(product_id=request.POST["pid"],question10="Excellent")
        good=question_answer.objects.filter(product_id=request.POST["pid"],question10="Good")
        satisfactory=question_answer.objects.filter(product_id=request.POST["pid"],question10="Satisfactory")
        poor=question_answer.objects.filter(product_id=request.POST["pid"],question10="Poor")
        verypoor=question_answer.objects.filter(product_id=request.POST["pid"],question10="Verypoor")
        excellent=len(excellent)
        good=len(good)
        satisfactory=len(satisfactory)
        poor=len(poor)
        verypoor=len(verypoor)
        positive=record_sentiment.objects.filter(product_id=request.POST["pid"],sentiment="positive")
        negative=record_sentiment.objects.filter(product_id=request.POST["pid"],sentiment="negative")
        neutral1=record_sentiment.objects.filter(product_id=request.POST["pid"],sentiment="neutral")
        negative=len(negative)
        positive=len(positive)
        neutral1=len(neutral1)
    elif(request.POST["ptype"]=="sports"):
        yes1=question_answer.objects.filter(product_id=request.POST["pid"],question1="yes")
        no1=question_answer.objects.filter(product_id=request.POST["pid"],question1="no")
        yes1=len(yes1)
        no1=len(no1)
        yes2=question_answer.objects.filter(product_id=request.POST["pid"],question2="yes")
        no2=question_answer.objects.filter(product_id=request.POST["pid"],question2="no")
        yes2=len(yes2)
        no2=len(no2)
        yes4=question_answer.objects.filter(product_id=request.POST["pid"],question4="yes")
        no4=question_answer.objects.filter(product_id=request.POST["pid"],question4="no")
        maybe4=question_answer.objects.filter(product_id=request.POST["pid"],question4="maybe")
        maybe4=len(maybe4)
        yes4=len(yes4)
        no4=len(no4)
        one=question_answer.objects.filter(product_id=request.POST["pid"],question5="1")
        two=question_answer.objects.filter(product_id=request.POST["pid"],question5="2")
        three=question_answer.objects.filter(product_id=request.POST["pid"],question5="3")
        four=question_answer.objects.filter(product_id=request.POST["pid"],question5="4")
        five=question_answer.objects.filter(product_id=request.POST["pid"],question5="5")
        one=len(one)
        two=len(two)
        three=len(three)
        four=len(four)
        five=len(five)
        yes9=question_answer.objects.filter(product_id=request.POST["pid"],question9="yes")
        no9=question_answer.objects.filter(product_id=request.POST["pid"],question9="no")
        yes9=len(yes9)
        no9=len(no9)
        excellent=question_answer.objects.filter(product_id=request.POST["pid"],question10="Excellent")
        good=question_answer.objects.filter(product_id=request.POST["pid"],question10="Good")
        satisfactory=question_answer.objects.filter(product_id=request.POST["pid"],question10="Satisfactory")
        poor=question_answer.objects.filter(product_id=request.POST["pid"],question10="Poor")
        verypoor=question_answer.objects.filter(product_id=request.POST["pid"],question10="Verypoor")
        excellent=len(excellent)
        good=len(good)
        satisfactory=len(satisfactory)
        poor=len(poor)
        verypoor=len(verypoor)
        positive=record_sentiment.objects.filter(product_id=request.POST["pid"],sentiment="positive")
        negative=record_sentiment.objects.filter(product_id=request.POST["pid"],sentiment="negative")
        neutral1=record_sentiment.objects.filter(product_id=request.POST["pid"],sentiment="neutral")
        negative=len(negative)
        positive=len(positive)
        neutral1=len(neutral1)

    total=yes1+no1
    if(request.POST["ptype"]=="food"):
        return render(request,'company_visualize.html',{'total':total,'ptype':request.POST['ptype'],'pname':request.POST['pname'],'neutral':neutral,'sad':sad,'angry':angry,'happy':happy,'suprise':suprise,'yes1':yes1,'no1':no1,'yes2':yes2,'no2':no2,'yes3':yes3,'no3':no3,'yes5':yes5,'no5':no5,'maybe5':maybe5,'yes8':yes8,'no8':no8,'excellent':excellent,'good':good,'satisfactory':satisfactory,'poor':poor,'verypoor':verypoor,'positive':positive,'negative':negative,'neutral1':neutral1})
    elif(request.POST["ptype"]=="vechile"):
        return render(request,'company_visualize.html',{'total':total,'ptype':request.POST['ptype'],'pname':request.POST['pname'],'neutral':neutral,'sad':sad,'angry':angry,'happy':happy,'suprise':suprise,'yes1':yes1,'no1':no1,'yes2':yes2,'no2':no2,'yes3':yes3,'no3':no3,'yes4':yes4,'no4':no4,'yes5':yes5,'no5':no5,'maybe5':maybe5,'yes7':yes7,'no7':no7,'maybe7':maybe7,'yes9':yes9,'no9':no9,'excellent':excellent,'good':good,'satisfactory':satisfactory,'poor':poor,'verypoor':verypoor,'positive':positive,'negative':negative,'neutral1':neutral1})
    elif(request.POST["ptype"]=="clothes"):
        return render(request,'company_visualize.html',{'total':total,'ptype':request.POST['ptype'],'pname':request.POST['pname'],'neutral':neutral,'sad':sad,'angry':angry,'happy':happy,'suprise':suprise,'yes1':yes1,'no1':no1,'yes2':yes2,'no2':no2,'yes3':yes3,'no3':no3,'yes4':yes4,'no4':no4,'yes7':yes7,'no7':no7,'summer':summer,'winter':winter,'excellent':excellent,'good':good,'satisfactory':satisfactory,'poor':poor,'verypoor':verypoor,'positive':positive,'negative':negative,'neutral1':neutral1})
    elif(request.POST["ptype"]=="tech"):
        return render(request,'company_visualize.html',{'total':total,'ptype':request.POST['ptype'],'pname':request.POST['pname'],'neutral':neutral,'sad':sad,'angry':angry,'happy':happy,'suprise':suprise,'yes1':yes1,'no1':no1,'yes2':yes2,'no2':no2,'design':design,'features':features,'both':both,'yes4':yes4,'no4':no4,'yes7':yes7,'no7':no7,'excellent':excellent,'good':good,'satisfactory':satisfactory,'poor':poor,'verypoor':verypoor,'positive':positive,'negative':negative,'neutral1':neutral1,'one':one,'two':two,'three':three,'four':four,'five':five})
    elif(request.POST["ptype"]=="sports"):
        return render(request,'company_visualize.html',{'total':total,'ptype':request.POST['ptype'],'pname':request.POST['pname'],'neutral':neutral,'sad':sad,'angry':angry,'happy':happy,'suprise':suprise,'yes1':yes1,'no1':no1,'yes2':yes2,'no2':no2,'yes4':yes4,'no4':no4,'maybe4':maybe4,'yes9':yes9,'no9':no9,'excellent':excellent,'good':good,'satisfactory':satisfactory,'poor':poor,'verypoor':verypoor,'positive':positive,'negative':negative,'neutral1':neutral1,'one':one,'two':two,'three':three,'four':four,'five':five})

    return render(request,'company_visualize.html')

def company_reports(request):
    p=Image_upload.objects.filter(company_id=request.session['user'])

    return render(request,'company_reports.html',{'p':p})

def questions_answer(request):

    p=question_answer.objects.create(product_id=request.POST["pid"],question1=request.POST["1"],question2=request.POST["2"],question3=request.POST["3"],question4=request.POST["4"],question5=request.POST["5"],question6=request.POST["6"],question7=request.POST["7"],question8=request.POST["8"],question9=request.POST["9"],question10=request.POST["10"])
    k=review_done.objects.create(product_id=request.POST["pid"],reviewer_id=request.session['user'])
    o=Image_upload.objects.filter(id=request.POST["pid"])
    print(o[0].title)
    print(o[0].company_id)
    t=company_info.objects.filter(id=o[0].company_id)
    q=review_incentive.objects.create(product_id=o[0].id,product_name=o[0].title,reviewer_id=request.session['user'],coupon_code=random.randint(1000,9999),company_name=t[0].name)
    if request.POST['ptype']=="food":
        txt=request.POST["4"]
        txt+=request.POST["7"]
        txt+=request.POST["9"]

    elif request.POST["ptype"]=="vechile":
        txt=request.POST["6"]
        txt+=request.POST["8"]

    elif request.POST["ptype"]=="clothes":
        txt=request.POST["6"]
        txt+=request.POST["9"]
    elif request.POST["ptype"]=="sports":
        txt=request.POST["6"]
        txt+=request.POST["7"]
        txt+=request.POST["8"]
    elif request.POST["ptype"]=="tech":
        txt=request.POST["8"]
        txt+=request.POST["9"]
    h=TextBlob(txt).sentiment.polarity
    if h>0.2:
        sentiment="positive"
    elif h<-0.2:
        sentiment="negative"
    else:
        sentiment="neutral"
    p= record_reaction.objects.create(reaction=request.POST['reaction'],product_id=request.POST['pid'],reviewer_id=request.session['user'])


    u=record_sentiment.objects.create(product_id=request.POST["pid"],reviewer_id=request.session['user'],sentiment=sentiment)
    return render(request,'reviewer_thanks.html')
def overview_reviewer(request):

    ii=review_incentive.objects.filter(reviewer_id=request.session['user'])

    oo=review_done.objects.filter(reviewer_id=request.session['user'])
    z=[]
    for i in oo:
        z.append(i.product_id)
    o=Image_upload.objects.exclude(id__in=z)
    k=[]
    for x in o:
        splitname=(x.product_type).split(",")
        for xx in splitname:
            if xx in request.session['typ']:
                k.append(x)
                break
    print(o)
    return render(request,'reviewer_dash.html',{'review_no':len(k),'incentive_no':len(ii)})

def overview_company(request):
    o=Image_upload.objects.filter(company_id=request.session['user'])
    u=[]

    for x in o:
        u.append(x.id)

    q=question_answer.objects.filter(product_id__in=u)

    return render(request,'company_dash.html',{'total_review':len(q),'total_product':len(o)})
def incentives(request):
    p=review_incentive.objects.filter(reviewer_id=request.session['user'])
    return render(request,'incentives_reviewer.html',{'p':p})

def company_profile(request):
    p=0
    message=""
    if(request.method=='POST'):
        tt=company_info.objects.filter(id=request.session['user'])
        for ttt in tt:
            if ttt.password == request.POST['pwd']:
                ttt.password=request.POST['repwd']
                ttt.save()
                p=1
                message="Password Sucessfully Changed"
            else:
                message="The current password is wrong"
                p=2
    o=Image_upload.objects.filter(company_id=request.session['user'])
    u=[]

    for x in o:
        u.append(x.id)

    q=question_answer.objects.filter(product_id__in=u)

    return render(request,'company_profile.html',{'total_review':len(q),'total_product':len(o),'p':p,'message':message})

def reviewer_profile(request):
    p=0
    message=""
    if(request.method=='POST'):
        tt=reviewer_info.objects.filter(id=request.session['user'])
        for ttt in tt:
            if ttt.password == request.POST['pwd']:
                ttt.password=request.POST['repwd']
                ttt.save()
                p=1
                message="Password Sucessfully Changed"
            else:
                message="The current password you entered is wrong"
                p=2
    ii=review_incentive.objects.filter(reviewer_id=request.session['user'])
    oo=review_done.objects.filter(reviewer_id=request.session['user'])
    z=[]
    ttt=reviewer_info.objects.filter(id=request.session['user'])

    for i in oo:
        z.append(i.product_id)
    o=Image_upload.objects.exclude(id__in=z)

    return render(request,'reviewer_profile.html',{'o':len(o),'ii':len(ii),'message':message,'p':p,'type':ttt[0].product_type})

def product_added(request):
    return render(request,'s.html')
