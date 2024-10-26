from django.db import models

# Create your models here.
from django.db import models

class InvestmentBasket(models.Model):
    image = models.URLField(max_length=200,blank=True,null=True)
    name = models.CharField(max_length=255,blank=True,null=True)
    volatility = models.CharField(max_length=255,blank=True,null=True)
    cagr = models.FloatField(help_text="Compound Annual Growth Rate",blank=True,null=True)
    basket_description = models.TextField(blank=True,null=True)
    manager_description = models.TextField(blank=True,null=True)
    minimum_investment_amount = models.FloatField(blank=True,null=True)

    def __str__(self):
        return self.name

class Holding(models.Model):
    basket = models.ForeignKey(InvestmentBasket, related_name='holdings', on_delete=models.CASCADE)
    asset_name = models.CharField(max_length=255,blank=True,null=True)
    distribution_percentage = models.FloatField(blank=True,null=True)

    def __str__(self):
        return f"{self.asset_name} - {self.distribution_percentage}%"
