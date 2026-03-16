from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

@app.route('/', methods=['GET','POST'])
def home():
    predict_pipeline=PredictPipeline()
    model_info = predict_pipeline.get_model_info()
    all_models = model_info.get('all_models', {})
    # Sort models by score (descending)
    all_models_sorted = dict(sorted(all_models.items(), key=lambda x: x[1], reverse=True))
    
    if request.method=='GET':
        return render_template('home.html',
                              model_name=model_info.get('best_model_name'), 
                              model_score=model_info.get('best_model_score'),
                              all_models=all_models_sorted)
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        results=predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results[0],
                              model_name=model_info.get('best_model_name'), 
                              model_score=model_info.get('best_model_score'),
                              all_models=all_models_sorted)
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)   


