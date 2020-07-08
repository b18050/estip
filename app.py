import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
memoryfree = pickle.load(open('assets/memoryfree.pkl', 'rb'))
memoryUsed = pickle.load(open('assets/memoryUsed.pkl', 'rb'))
CPUUtil    = pickle.load(open('assets/processor.pkl', 'rb')) 

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    prediction1 = memoryfree.predict(final_features)
    prediction2 = memoryUsed.predict(final_features)
    
    prediction1 = abs(prediction1/10**8)
    prediction2 = abs(prediction2/10**8)
    
    prediction = prediction1 + prediction2
    output1 = int(prediction[0])
    
    prediction3 = CPUUtil.predict(final_features)
    output2 = int(abs(prediction3[0]))/10
    

    return render_template('index.html', memory_text='Memory      at least {} GB'.format(output1), processor_text='Processor     more than {} GHz'.format(output2))



if __name__ == "__main__":
    app.run(debug=True)