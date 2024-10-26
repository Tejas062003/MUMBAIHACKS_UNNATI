from django.contrib import admin
from django.urls import path
from .views import *

urlpatterns = [
    path('stock_analysis/', stock_analysis, name='stock_analysis'),
    path('investment_baskets/', fetch_investment_baskets, name='fetch_investment_baskets'),
    path('basket/<int:basket_id>/', investment_basket_detail, name='basket_detail'),
    path('playground/', playground, name='optimize_portfolio'),
    path('optimize_portfolio', optimize_portfolio, name='optimize_portfolio'),
    path('ai_bot', ai_bot_view, name='ai_bot'),
    path('simulate/', simulate, name='simulate'),
    path('simulator/', home, name='simulator'),
    path('financial_mentor/',financial_mentor, name='financial_mentor'),
        path('better_basket/', better_basket, name='better_basket'),

        # path('monte-carlo-simulation/', monte_carlo_simulation, name='monte_carlo_simulation'),
]
