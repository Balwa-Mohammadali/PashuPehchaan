
# from django.urls import path
# from . import views

# urlpatterns = [
#     path('', views.index, name='index'),
#     path('predict/', views.predict, name='predict'),
# ]

from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("about/", views.about, name="about"),
    path("predict/", views.predict_page, name="predict_page"),
    path("predict/run/", views.predict, name="predict"),
]
