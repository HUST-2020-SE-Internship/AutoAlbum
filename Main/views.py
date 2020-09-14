from django.shortcuts import render, redirect
from django.http import HttpResponse
import json
import base64
import numpy as np
import cv2

import sys

# sys.path.append("")
from ImageNet import VGGLib
from ImageNet.VGGLib import VGGNet


# Create your views here.
def mainPage(request):
    if not request.user.is_authenticated:
        login_msg = "请登陆账号访问"
        return render(request, "login.html", locals())
    else:
        return render(request, "main.html")


def classifyImage(request):
    if request.method == "POST":
        image = request.POST.get("image")
        image = json.loads(image)
        image = str(image).split(';base64,')[1]
        image = base64.b64decode(image)
        image = np.fromstring(image, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        result = VGGLib.classifyImage(image)
    return HttpResponse(json.dumps(result))
