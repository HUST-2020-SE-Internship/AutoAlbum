from django.urls import re_path
from . import views

urlpatterns = [
    re_path(r'^$', views.loginPage),
    re_path(r'^loginAccount/$', views.loginAccount),
    re_path(r'^registAccount/$', views.registAccount),
    re_path(r'^logoutAccount/$', views.logoutAccount)
]