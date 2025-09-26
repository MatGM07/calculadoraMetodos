"""
URL configuration for metodos project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
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
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('secante/', views.secante, name='secante'),
    path('secanteResuelta/', views.secanteResuelta, name='secanteResuelta'),
    path('ajuste/', views.ajuste, name='ajuste'),
    path('ajusteResuelta/', views.ajusteResuelta, name='ajusteResuelta'),
    path('falsaPosicion/', views.falsaPosicion, name='falsaPosicion'),
    path('falsaPosicionResuelta/', views.falsaPosicionResuelta, name='falsaPosicionResuelta'),
    path('integracion/', views.integracion, name='integracion'),
    path('integracionResuelta/', views.integracionResuelta, name='integracionResuelta'),
    path('diferenciacion/', views.diferenciacion, name='diferenciacion'),
    path('diferenciacionResuelta/', views.diferenciacionResuelta, name='diferenciacionResuelta'),
    path('diferenciacionTabla/', views.diferenciacionTabla, name='diferenciacionTabla'),
    path('diferenciacionTablaResuelta/', views.diferenciacionTablaResuelta, name='diferenciacionTablaResuelta'),
    path('optimizacion/', views.optimizacion, name='optimizacion'),
    path('optimizacionResuelta/', views.optimizacionResuelta, name='optimizacionResuelta'),
    path('', views.index, name='index'),
]
