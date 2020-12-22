from django import forms
from django.forms.formsets import formset_factory
'''
Pregnancies: Number of times pregnant
Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
BloodPressure: Diastolic blood pressure (mm Hg)
SkinThickness: Triceps skin fold thickness (mm)
Insulin: 2-Hour serum insulin (mu U/ml)
BMI: Body mass index (weight in kg/(height in m)^2)
DiabetesPedigreeFunction: Diabetes pedigree function
Age: Age (years)

'''
class InputForm(forms.Form):
   # Text = forms.CharField(max_length=80, widget=forms.TextInput(attrs={'class': 'autocomplete'}))
    Pregnancies=forms.IntegerField(max_value=2000,min_value=0,
    	widget=forms.NumberInput(attrs={'class': 'form-control'}),
    	help_text='Number of times pregnant')
    Glucose=forms.FloatField(max_value=2000,min_value=0,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        help_text='Plasma glucose concentration a 2 hours in an oral glucose tolerance test')
    BloodPressure=forms.IntegerField(max_value=2000,min_value=0,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        help_text='Diastolic blood pressure (mm Hg)')
    SkinThickness=forms.IntegerField(max_value=2000,min_value=0,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        help_text='Triceps skin fold thickness (mm)')
    Insulin=forms.IntegerField(max_value=2000,min_value=0,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        help_text='2-Hour serum insulin (mu U/ml)')
    BMI=forms.FloatField(max_value=2000,min_value=0,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        help_text='Body mass index (weight in kg/(height in m)^2)')
    DiabetesPedigreeFunction=forms.FloatField(max_value=2000,min_value=0,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        help_text='Diabetes pedigree function')
    Age=forms.IntegerField(max_value=2000,min_value=0,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        help_text='Age (years)')






