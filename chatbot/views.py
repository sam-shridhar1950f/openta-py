from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
import openai
import logging
from django.conf import settings


from django.contrib import auth
from django.contrib.auth.models import User
from .models import Chat

from django.utils import timezone

import os
import openai
from scipy.spatial import distance
import plotly.express as px
from sklearn.cluster import KMeans
from umap import UMAP
import pinecone
import re
from transformers import pipeline
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
from .summary import summarize
import time
from .course import Course


fmt = getattr(settings, 'LOG_FORMAT', None)
lvl = getattr(settings, 'LOG_LEVEL', logging.DEBUG)

logging.basicConfig(format=fmt, level=lvl)
logging.debug("Logging started on %s for %s" %
              (logging.root.name, logging.getLevelName(lvl)))

# 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" #'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'
openai_api_key = "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"
openai.api_key = openai_api_key

pinecone.init(api_key="190dabf8-0c0b-4690-8733-0be7fcecdb34")

COURSE = ""
SCHOOL = ""
COURSE_BEFORE = ""

SET = False
course_selection = None


def ask_openai(message, query_type):
    # response = openai.ChatCompletion.create(
    #     model = "gpt-4",
    #     messages=[
    #         {"role": "system", "content": "You are an helpful assistant."},
    #         {"role": "user", "content": message},
    #     ]
    # )
    global SET
    global course_selection
    global COURSE_BEFORE
    if COURSE != COURSE_BEFORE:  # COURSE: Stanford COURSE_BEFORE: ""
        course_selection = Course(COURSE, SCHOOL, openai.api_key)
        # SET = True
        COURSE_BEFORE = COURSE

    response = course_selection.main(message, query_type)

    answer = response
    return answer

# Create your views here.


def chatbot(request):
    # tempUser= User.objects.create_user(username="anon", email="none", password="none")
    # tempUser.save()
    if request.user.is_anonymous:
        anonUser = User.objects.get(username='anon')
        request.user = User.objects.get(id=anonUser.id)
    chats = Chat.objects.filter(user=request.user)

    if request.method == 'POST':
        message = request.POST.get('message')
        query_type = request.POST.get('query_type')

        response = ask_openai(message, query_type)
        # if request.user.is_anonymous:
        #     anonUser = User.objects.get(username='anon')
        #     request.user=User.objects.get(id=anonUser.id)

        chat = Chat(user=request.user, message=message,
                    response=response, created_at=timezone.now())
        chat.save()
        return JsonResponse({'message': message, 'response': response})
    return render(request, 'chatbot.html', {'chats': chats, 'course': COURSE, 'school': SCHOOL})


def login(request):
    # tempUser= User.objects.create_user(username="anon", email="none", password="none")
    # tempUser.save()
    if request.user.is_anonymous:
        anonUser = User.objects.get(username='anon')
        request.user = User.objects.get(id=anonUser.id)
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        # if request.user.is_anonymous:
        #     anonUser = User.objects.get(username='anon')
        #     request.user=User.objects.get(id=anonUser.id)

        user = auth.authenticate(request, username=username, password=password)
        if user is not None:
            auth.login(request, user)
            return redirect('organizations')
        else:
            error_message = 'Invalid username or password'
            return render(request, 'login.html', {'error_message': error_message})
    else:
        return render(request, 'login.html')


def register(request):
    # tempUser= User.objects.create_user(username="anon", email="none", password="none")
    # tempUser.save()
    if request.user.is_anonymous:
        anonUser = User.objects.get(username='anon')
        request.user = User.objects.get(id=anonUser.id)
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password1 = request.POST['password1']
        password2 = request.POST['password2']

        # if request.user.is_anonymous:
        #     anonUser = User.objects.get(username='anon')
        #     request.user=User.objects.get(id=anonUser.id)

        if password1 == password2:
            try:
                user = User.objects.create_user(username, email, password1)
                user.save()
                auth.login(request, user)
                return redirect('organizations')
            except:
                error_message = 'Error creating account'
                return render(request, 'register.html', {'error_message': error_message})
        else:
            error_message = 'Password dont match'
            return render(request, 'register.html', {'error_message': error_message})
    return render(request, 'register.html')


def logout(request):
    auth.logout(request)
    return redirect('login')


@login_required
def organizations(request):
    logging.debug("Oh hai!")
    if request.method == 'POST':
        name = request.POST['Select']
        if name == "Stanford":
            return redirect('stanforddashboard')
        elif name == "Yale":
            return redirect('yaledashboard')
        elif name == "Cornell":
            return redirect('cornelldashboard')
        elif name == "IBM":
            return redirect('ibmdashboard')

        # return redirect('chatbot')
    return render(request, 'organizations.html')

# @login_required


def stanforddashboard(request):
    global COURSE
    global SCHOOL
    if request.method == 'POST':
        name = request.POST['Chat']
        name = name.split()
        print(name)
        COURSE = name[0]
        SCHOOL = name[2]
        print(name)

        return redirect('chatbot')
    return render(request, 'stanforddashboard.html')


def yaledashboard(request):
    global COURSE
    global SCHOOL
    if request.method == 'POST':
        name = request.POST['Chat']
        name = name.split()
        print(name)
        COURSE = name[0]
        SCHOOL = name[2]
        print(name)

        return redirect('chatbot')
    return render(request, 'yaledashboard.html')


def cornelldashboard(request):
    global COURSE
    global SCHOOL
    if request.method == 'POST':
        name = request.POST['Chat']
        name = name.split()
        print(name)
        COURSE = name[0]
        SCHOOL = name[2]
        print(name)

        return redirect('chatbot')
    return render(request, 'cornelldashboard.html')


def ibmdashboard(request):
    global COURSE
    global SCHOOL
    if request.method == 'POST':
        name = request.POST['Chat']
        name = name.split()
        print(name)
        COURSE = name[0]
        SCHOOL = name[2]
        print(name)

        return redirect('chatbot')
    return render(request, 'ibmdashboard.html')
