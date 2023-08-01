from django.shortcuts import render
import numpy as np
import pickle

# Create your views here.



def index(Request):
    # Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age 
    if Request.method == "POST":
        result = ""
   
        Pregnancies =   Request.POST.get("Pregnancies")
        Glucose =   Request.POST.get("Glucose")
        BloodPressure =   Request.POST.get("BloodPressure")
        SkinThickness =   Request.POST.get("SkinThickness")
        Insulin =   Request.POST.get("Insulin")
        BMI =   Request.POST.get("BMI")
        DiabetesPedigreeFunction =   Request.POST.get("DiabetesPedigreeFunction")
        Age =   Request.POST.get("Age")
      
      
        model = pickle.load(open('trained_model.sav', 'rb'))
        input_data = (Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age)
        input_data_as_numpy_array = np.asarray(input_data,dtype=np.float64)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        prediction = model.predict(input_data_reshaped)
        
        print(prediction)
        if(prediction[0]==0):
            result = "The person is not diabetic"
            print("The person is not diabetic")
        else:
            result = "The person is diabetic"
            print("The person is diabetic")
      
        context = {
            "result":result
        }
        return render(Request,"App/index.html",context)
      
    return render(Request,"App/index.html")