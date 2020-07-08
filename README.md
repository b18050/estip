##### Demo
[LINK To WebAppp](https://estipapp.herokuapp.com/)

## BNG Data Predictor
This is a demo project that predicts about the requirements of BNG device using Machine Learning Models deployed using Flask API.

### Prerequisites
You must have Scikit Learn, Pandas (for Machine Leraning Model) and Flask (for API) installed.

### Project Structure
This project has four major parts :
1. Models 
	1. memoryfree.py - This contains code for our Machine Learning model to predict memory not in use or free Memory based on datasets in 'data.csv' file.
	2. memoryused.py - This conatins code for our Machine Learning model to predict memory currently in use based on datasets in 'data.csv' file.
	3. processor.py - This contains code for our Machine Learning model to predict processor requirements of BNG device based on datasets in 'data.csv' file.
2. app.py - This contains Flask APIs that receives BNG data details through GUI or API calls, computes the precited value based on our models and returns it.
3. templates - This folder contains the HTML template to allow user to enter BNG requirements and displays the predicted value specifics required by BNG device.

### Running the project
1. Ensure that you are in the project home directory. Create the machine learning model by running below command -
```
python memoryfree.py
python memoryused.py
python processor.py
```
This would create a serialized version of our models into files memoryfree.pkl, memoryused.pkl , processor.pkl

2. Run app.py using below command to start Flask API
```
python app.py
```
By default, flask will run on port 5000.

3. Navigate to URL http://localhost:5000

You should be able to view the homepage as below :
![alt text](/assets/image1.jpg)

Enter valid numerical values in all 3 input boxes and hit Predict.

If everything goes well, you should  be able to see the predcited vaule on the HTML page!
[LINK To WebAppp](https://estipapp.herokuapp.com/)

4. Inspiration
[link](https://github.com/krishnaik06/Deployment-flask)

