from django.urls import path
from . import views

from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.login, name='login'),
    path('register', views.register, name='register'),
    path('chatbot', views.chatbot, name='chatbot'),
    path('stanforddashboard', views.stanforddashboard, name='stanforddashboard'),
    path('yaledashboard', views.yaledashboard, name='yaledashboard'),
    path('cornelldashboard', views.cornelldashboard, name='cornelldashboard'),
    path('ibmdashboard', views.ibmdashboard, name='ibmdashboard'),
    path('organizations', views.organizations, name='organizations'),
    path('logout', views.logout, name='logout'),
    # path('', views.chatbot, name='chatbot'),
    # path('login', views.login, name='login'),
    # path('register', views.register, name='register'),
    # path('logout', views.logout, name='logout'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)