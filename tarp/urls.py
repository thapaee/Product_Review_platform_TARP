"""hci URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from tarp import views
import reviewer.views
urlpatterns = [
    path("admin/",admin.site.urls),
    path("",views.homepage,name="home"),
    path("services/",views.servicepage,name="services"),
    path("choose/",reviewer.views.choose,name="choose"),
    path("signup_choose/",reviewer.views.signup_landing,name="signup_landing"),
    path("signup_review/",reviewer.views.reviewer_signup,name="reviewer_signup"),
    path("signup_company/",reviewer.views.company_signup,name="company_signup"),
    path("choose/company",reviewer.views.company_login,name="company_login"),
    path("choose/reviewer",reviewer.views.company_login,name="reviewer_login"),
    path('t/',reviewer.views.cap,name='act'),
    path('review/',reviewer.views.review,name='ac'),
    path('/start/',views.start,name='start'),
    path('st/',views.store,name='store'),
    path('added/',reviewer.views.add_to_model,name='add_to_model'),
    path('company_dash/',reviewer.views.company_dash,name='company_dash'),
    path('company_profile/',reviewer.views.company_profile,name='company_profile'),
    path('reviewer_profile/',reviewer.views.reviewer_profile,name='reviewer_profile'),
    path('/add_product/',reviewer.views.add_product,name="upload"),
    path('/add_produc/',reviewer.views.add_final,name="uploadd"),
    path('view_product/',reviewer.views.view_product,name="view_products"),
    path('signedout/',views.signout,name="signout"),
    path('login_reviewer/',reviewer.views.reviewer_login,name="reviewer_login"),
    path('login_company/',reviewer.views.company_login,name="company_login"),
    path('record_review/',reviewer.views.record_expression,name="record_expression"),
    path('stats/',reviewer.views.company_visualize,name="company_visualize"),
    path('reports/',reviewer.views.company_reports,name="company_reports"),
    path('overview_reviewer/',reviewer.views.overview_reviewer,name="overview_reviewer"),
    path('overview_company/',reviewer.views.overview_company,name="overview_company"),
    path('thanks/',reviewer.views.questions_answer,name="reviewer_thanks"),
    path('incentives/',reviewer.views.incentives,name='incentives'),
    path('yu/',reviewer.views.sucess,name='s'),
    path('read_description/',reviewer.views.read_description,name='read_description'),
    path('sucess/',reviewer.views.product_added,name='added')
]
if settings.DEBUG:
        urlpatterns += static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)
