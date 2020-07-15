import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template,url_for, redirect
import pickle
from sklearn.feature_extraction.text import HashingVectorizer
import re
import os
import nltk
import random
nltk.download('stopwords')
from nltk.corpus import stopwords

app = Flask(__name__)
memoryfree = pickle.load(open('assets/memoryfree.pkl', 'rb'))
memoryUsed = pickle.load(open('assets/memoryUsed.pkl', 'rb'))
CPUUtil    = pickle.load(open('assets/processor.pkl', 'rb')) 
stop = pickle.load(open('assets/stopwords.pkl', 'rb'))
classifier = pickle.load(open('assets/classifier.pkl', 'rb')) 
churn_model = pickle.load(open('assets/model.pkl','rb'))
churn_columns = pickle.load(open('assets/model_columns.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/customer/')
def sentiment():
    '''
    For rendering cutomer page 
    '''
    # IMAGE_FOLDER = os.path.join('static', 'images')
    # app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER

    # full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'all.jpg')

    return render_template('customer.html' )

@app.route('/about/')
def about():
    '''
    For rendering about page 
    '''
    return render_template('about.html')

@app.route('/abouttelecom/')
def abouttelecom():
    '''
    For rendering about page 
    '''
    return render_template('abouttelecom.html')

@app.route('/aboutchurn/')
def aboutchurn():
    '''
    For rendering about page 
    '''
    return render_template('aboutchurn.html')

@app.route('/bng/')
def bng():
    '''
    For rendering about page 
    '''
    return render_template('bng.html')

@app.route('/bngpredict',methods=['POST'])
def bngpredict():
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
    

    return render_template('bng.html', memory_text='Memory      at least {} GB'.format(output1), processor_text='Processor     more than {} GHz'.format(output2))

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) +\
        ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stopwords.words('english')]
    # this will return the cleaned text
    return tokenized

def random_bool():
    return random_number()
 
def random_number(low=0, high=1):
    return random.randint(low,high)

def generate_data():
    internetServices = ['DSL', 'Fiber optic', 'No']
    contracts = ['Month-to-month', 'One year', 'Two year']
    paymentMethods = ['Electronic check', 'Mailed check', 'Bank transfer (automatic)','Credit card (automatic)']
 
    random_data = {
            'name':'customer',
            'Partner': random_bool(),
            'Dependents': random_bool(),
            'tenure': random_number(0,50),
            'PhoneService': random_bool(),
            'MultipleLines': random_number(-1),
            'InternetService': random.choice(internetServices),
            'OnlineSecurity': random_number(-1),
            'OnlineBackup': random_number(-1),
            'DeviceProtection': random_number(-1),
            'TechSupport': random_number(-1),
            'StreamingTV': random_number(-1),
            'StreamingMovies': random_number(-1),
            'Contract': random.choice(contracts),
            'PaperlessBilling': random_bool(),
            'PaymentMethod': random.choice(paymentMethods)
        }
    return random_data

@app.route('/churnpredict', methods=['POST'])
def churnpredict():
    random_user_data = generate_data()
    query = pd.get_dummies(pd.DataFrame(random_user_data, index=[0]))
    query = query.reindex(columns=churn_columns, fill_value=0)
 
    #return prediction as probability in percent
    prediction = round(churn_model.predict_proba(query)[:,1][0], 2)* 100
    return render_template('churn.html' , proba='Customer has a chance of churn with  {} '.format(int(prediction)))

@app.route('/churn/')
def churn():
    '''
    For rendering churn page 
    '''
    return render_template('churn.html')



@app.route('/sentimentpredict', methods=['POST'])
def sentimentpredict():

    vect = HashingVectorizer( decode_error='ignore' ,n_features=2**21 ,preprocessor=None,tokenizer=tokenizer )

    # get text from the incoming request (submitted on predict button click)
    text = request.form['input_text']

    # convert text to model input vector
    X = vect.transform([text])

    # use classifier's predict method to get prediction
    y = classifier.predict(X)

    # store model input and output
    # model_input = text
    sentiment_output = y[0]
    proba = np.max(classifier.predict_proba(X))
    proba = round(proba,2)

    IMAGE_FOLDER = os.path.join('static', 'images')
    app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER
    if(sentiment_output == 1 and proba >= 0.9):
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'excellent.jpg')
    elif(proba >= 0.8 and proba < 0.9):
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'love.jpg')
    elif(proba >= 0.7 and proba < 0.8):
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'ok.jpg')
    elif(proba >= 0.6 and proba < 0.7):
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'normal.jpg')
    elif(proba >= 0.5 and proba < 0.6):
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'bad.png')
    elif(proba >= 0.4 and proba < 0.5):
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'awful.jpg')
    elif( proba >= 0.3 and proba < 0.4):
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'pain.jpg')
    elif(proba >= 0.2 and proba < 0.3):
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'cry.jpg')
    elif( proba >= 0.1 and proba < 0.2):
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'sad.jpg')
    elif( proba >= 0.0 and proba < 0.1):
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'terrible.jpg')
    else:
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'ok.jpg')

    proba = round(proba,1) * 10
    
    return render_template('customer.html' , user_image = full_filename, proba='Rating {} (out of 10)  '.format(proba))
    


if __name__ == "__main__":
    app.run(debug=True)