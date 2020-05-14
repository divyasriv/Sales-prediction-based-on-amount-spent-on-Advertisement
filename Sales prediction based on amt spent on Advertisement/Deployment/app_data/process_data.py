# -*- coding: utf-8 -*-
"""
Created on Tue May 12 14:07:04 2020

@author: GSLP0676
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 12 13:19:50 2020

@author: GSLP0676
"""
from sklearn.externals import joblib
import numpy as np
import pandas as pd

class PassPipeline:
    def __init__(self):
        self.data = ""
        RFRmodel = open('app_data/RFReg_model.ml','rb')
        self.RFRmodel = joblib.load(RFRmodel)
        XGBmodel = open('app_data/XGBReg_model.ml','rb')
        self.XGBmodel = joblib.load(XGBmodel)
        LGBMmodel = open('app_data/LGBMReg_model.ml','rb')
        self.LGBMmodel = joblib.load(LGBMmodel)
        
    def pass_data(self,data):
        self.data = data     
      
    def get_data_for_model(self):
        amt_newspaper=self.data['Newspaper']
        amt_radio=self.data['Radio']
        amt_tv=self.data['TV']
        inp_arr=[amt_tv,amt_newspaper,amt_radio]
        data = np.array(inp_arr)
        data = data.astype(np.float).reshape(1,-1)
        return data
    
    def get_data_for_model1(self):
        amt_newspaper=float(self.data['Newspaper'])
        amt_radio=float(self.data['Radio'])
        amt_tv=float(self.data['TV'])
        data = pd.DataFrame({
                'TV': [amt_tv],
                'newspaper': [amt_newspaper],
                'radio': [amt_radio]                
                })
        return data
   
    def get_RFRprediction(self,model_data):
        RFRprediction = self.RFRmodel.predict(model_data)
        return RFRprediction
    def get_XGBprediction(self,model_data):
        XGBprediction = self.XGBmodel.predict(model_data)
        return XGBprediction
    def get_LGBMprediction(self,model_data):
        LGBMprediction = self.LGBMmodel.predict(model_data)
        return LGBMprediction
       
        
    

