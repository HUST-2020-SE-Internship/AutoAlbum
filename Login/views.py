from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.http import HttpResponse
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.shortcuts import redirect


# Create your views here.
def hello(request):
    if not request.user.is_authenticated:
        login_msg = "请登陆账号访问"
        return render(request, "login.html", locals())
    else:
        # return render(request, "hello.html")
        return redirect("/main")


def loginPage(request):
    return render(request, "login.html")


def loginAccount(request):
    if request.method == 'POST':
        username = request.POST.get("username")
        password = request.POST.get("password")
        print("username: ", username, "password: ", password)
        user = authenticate(username=username, password=password)
        if user:
            print("=== login success")
            login(request, user)
            return redirect("/")
        else:
            print("=== login fail")
            login_msg = "登录失败，请重试"
            return render(request, "login.html", locals())
    return render(request, "login.html")


def registAccount(request):
    if request.method == 'POST':
        username = request.POST.get("username")
        password = request.POST.get("password")
        password_confirm = request.POST.get("password_confirm")

        if username == '' or password == '' or password_confirm == '':
            print("=== info is None")
            login_msg = "信息不能为空"
            return render(request, "login.html", locals())
        elif password != password_confirm:
            login_msg = "两次输入密码不同"
            print("=== password not the same", password, password_confirm)
            return render(request, "login.html", locals())

        cuser = User.objects.create_user(username=username, password=password)
        cuser.save()
        print("=== regist success")
        login_msg = "注册成功，请登录"
        return render(request, "login.html", locals())


def logoutAccount(request):
    logout(request)
    return redirect("/login")
