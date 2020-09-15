from django.shortcuts import render, redirect
from django.http import HttpResponse
import json
import base64
import numpy as np
import cv2
from ImageNet import VGGLib
import os


# Create your views here.
def mainPage(request):
    if not request.user.is_authenticated:
        login_msg = "请登陆账号访问"
        return render(request, "login.html", locals())
    else:
        return render(request, "main.html")


def classifyImage(request):
    image = request.POST.get("image")
    image = json.loads(image)
    image = str(image).split(';base64,')[1]
    image = base64.b64decode(image)
    image = np.fromstring(image, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    result = VGGLib.classifyImage(image)
    return HttpResponse(json.dumps(result))


def saveImage(request):
    typeName = request.POST.get("typeName")
    image = request.POST.get("image")
    fileName = hash(image)
    image = json.loads(image)
    image = str(image).split(';base64,')[1]
    image = base64.b64decode(image)

    path = "media/" + request.user.username + "/" + typeName
    print(path, os.path.exists(path))
    if not os.path.exists(path):
        os.makedirs(path)

    filePath = os.path.join(path, str(fileName) + ".jpg")
    file = open(filePath, "wb")
    file.write(image)
    file.close()

    print("save => ", filePath)
    return HttpResponse("success")
