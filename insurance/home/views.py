from django.shortcuts import render, HttpResponse
import numpy as np
import pickle

with open('static/rf_pred.pkl', 'rb') as file:
    model = pickle.load(file)

# Create your views here.
def home(request):
    # return HttpResponse("This is the home page") 
    return render(request, 'index.html')

def about(request):
    # return HttpResponse("This is the about page") 
    return render(request, 'about.html')

def contact(request):
    # return HttpResponse("This is the contact page") 
    return render(request, 'contact.html')

def login(request):
    # return HttpResponse("This is the login page") 
    return render(request, 'login.html')

def registration(request):
    # return HttpResponse("This is the registration page") 
    return render(request, 'registration.html')

def prediction(request):
    # return HttpResponse("This is the prediction page")
    if request.method == "POST":
        # print("enter into the post request")
        age = int(request.POST.get('age'))
        sex = int(request.POST.get('sex'))
        bmi = float(request.POST.get('bmi'))
        children = int(request.POST.get('children'))
        smoker = int(request.POST.get('smoker'))
        region = int(request.POST.get('region'))

        encoded_region = np.zeros(4)  # Assuming there are 4 regions
        encoded_region[int(region) - 1] = 1

        # Now, you can concatenate all the features and send it to your model for prediction
        features = [age, sex, bmi, children, smoker] + list(encoded_region)

        # Reshape features to match the input shape expected by your model
        features = np.array(features).reshape(1, -1)

        # Call your machine learning model to make predictions
        prediction = model.predict(features)

        output = {
            "output": prediction
        }

        return render(request, 'prediction.html', output)

    else:
        return render(request, 'prediction.html')