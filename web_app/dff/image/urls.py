from django.urls import path

from . import views

urlpatterns = [
    path('', views.detect, name='index'),
]