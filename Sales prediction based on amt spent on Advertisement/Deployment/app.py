from flask import Flask,render_template,request
from app_data import process_data

app = Flask(__name__)
pipeline_object = process_data.PassPipeline()

@app.route('/')
def homePage():
    return render_template("index.html")

@app.route('/predictionPage',methods=['POST'])
def predictionPage():
    if request.method == 'POST':
        data=request.form
        pipeline_object.pass_data(data)
        model_data=pipeline_object.get_data_for_model1()
        predicted_value_RFR = pipeline_object.get_RFRprediction(model_data)
        predicted_value_XGB = pipeline_object.get_XGBprediction(model_data)
        predicted_value_LGBM = pipeline_object.get_LGBMprediction(model_data)
        predicted_value=[predicted_value_RFR,predicted_value_XGB,predicted_value_LGBM]
        return render_template("results.html",data=predicted_value)

if __name__ == "__main__":
    app.run(debug=True)
       