import json
import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from sklearn.ensemble import RandomForestRegressor
import os

# Get the path to the current directory
base_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the CSV file
csv_path = os.path.join(base_dir, '..', 'Position_Salaries.csv')

# Load and train the model once when the server starts
def train_model():
    try:
        dataset = pd.read_csv(csv_path)
        X = dataset.iloc[:, 1:-1].values
        y = dataset.iloc[:, -1].values
        regressor = RandomForestRegressor(n_estimators=10, random_state=0)
        regressor.fit(X, y)
        return regressor
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found. Please ensure it is in the project's root directory.")
        return None

regressor = train_model()

# This view renders the main HTML page.
def index(request):
    return render(request, 'predictor/index.html')

# This view handles the prediction request from the frontend.
@csrf_exempt
def predict_salary(request):
    if request.method == 'POST':
        if not regressor:
            return JsonResponse({'error': 'Model not trained. File not found.'}, status=500)
        
        try:
            data = json.loads(request.body)
            level = data.get('level')
            if level is None:
                return JsonResponse({'error': 'Level not provided.'}, status=400)
            
            # Reshape the input for the model
            prediction = regressor.predict([[level]])
            
            # Return the prediction as a JSON response
            return JsonResponse({'salary': prediction[0]})
        
        except (json.JSONDecodeError, KeyError, ValueError):
            return JsonResponse({'error': 'Invalid request body.'}, status=400)
    
    return JsonResponse({'error': 'Invalid request method.'}, status=405)