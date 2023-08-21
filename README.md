# Deployment of Salary Prediction in Data-Related Jobs Using Machine Learning Regression

The files here are used to deploy a salary prediction model from [this repository](https://github.com/wandalistathea/code-prediksi-gaji-profesi-data/tree/main) in the form of a web app.

The web app can be accessed in [prediksi-gaji-profesi-data.herokuapp.com](https://prediksi-gaji-profesi-data.herokuapp.com/) actually, but no longer now. So [here](https://drive.google.com/file/d/12NY84jRMX-ARWeOhNn8no9AITNGdED11/view?usp=sharing) is the screen recording of that web app result on my local page.

## How to Use
- **“static” folder:** Contains the CSS files and images
- **"templates" folder:** Contains HTML files for both the main page of the web and the prediction result page
- **"requirements.txt":** Contains the required Python packages

	Use this command in the terminal to automatically install all the packages

		pip install -r requirements.txt
  
- **"model_decision_tree.py":** Code for processing the data until saving the model (pickle form)

  Running this code to get the "model_decisiontree_smote.pkl" as the output

		python model_decision_tree.py
  
- **"app.py":** Flask code to unpickle the model and make predictions based on the input data

 	This command will provide the localhost address where the web app will appear

		python app.py
