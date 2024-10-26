from django.db import models

# Create your models here.
from django.db import models

class InvestmentOptimization(models.Model):
    LIFESTYLE_RISK_CHOICES = [
        ('low', 'Low'),
        ('mid', 'Mid'),
        ('high', 'High'),
    ]
    
    name = models.CharField(max_length=255)
    email = models.EmailField()
    lifestyle_risk = models.CharField(max_length=4, choices=LIFESTYLE_RISK_CHOICES)
    expected_annual_roi = models.IntegerField()
    monthly_investment = models.FloatField()
    current_age = models.IntegerField()
    retirement_age = models.IntegerField()
    annual_salary = models.FloatField()
    monthly_expenses = models.FloatField()
    savings = models.FloatField()
    principal_amount = models.FloatField()
    investment_period = models.IntegerField()

    def __str__(self):
        return f"{self.name} - {self.email}"