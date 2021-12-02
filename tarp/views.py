from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
def homepage(request):
    return render(request,'home.html')

def servicepage(request):
    return render(request,'services.html')

def start(request):
    return render(request,'start.html')

def store(request):
    if request.method == 'POST':
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
    return render(request,'fileupload.html')

def signout(request):
    del request.session['user']
    return render(request,'home.html')
