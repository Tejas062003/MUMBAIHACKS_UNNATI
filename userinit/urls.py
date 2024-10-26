from django.contrib import admin
from django.urls import path
from .views import *

urlpatterns = [
    path('calculator/', index, name='index'),
    path('calculate/', calculate_portfolio, name='calculate_portfolio'),
    path('portfolio/', portfolio_view, name='portfolio'),
    path('optimize-investment/', optimize_investment, name='optimize_investment'),
    path('base/', base, name='base'),
    path('form/', form, name='form'),
    path('', optimize_investment1, name='optimize_investment1'),

    # path('monte-carlo-simulation/', monte_carlo_simulation, name='monte_carlo_simulation'),



]
