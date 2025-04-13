#bhai code p focus matt krna ... bss logic and kaise kya ho rha h ... kyu ho rha h ... uspe focus krna ... 
#step by step code likhte rehna ... 

from flask import Flask, render_template, request
import mlflow
from preprocessing_utility import normalize_text
import dagshub
import pickle




mlflow.set_tracking_uri('https://dagshub.com/anujahlawat.ds/08-learning-mlops-ci.mlflow')
dagshub.init(repo_owner='anujahlawat.ds', repo_name='08-learning-mlops-ci', mlflow=True)




app = Flask(__name__)               #app is flask class ka object and __name__ is magic variable 




#step 01 :- load model from model registry ... 
def get_latest_model_version(model_name):            #model ka latest version fetch kr lega
    client = mlflow.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["Production"])
    if not latest_version:
        latest_version = client.get_latest_versions(model_name, stages=["None"])
    return latest_version[0].version if latest_version else None
#yha aapko 2 cheeje btani h ... aapke model ka naam kya h and aapko konsa version chahiye ... 
model_name = "my_model"         #mlflow m jake dekho ... ki aap konsa model load krana chahte ho ... 
#model_version = 1            #jo version production m h vhi daalenge na ... 
model_version = get_latest_model_version(model_name)
#note :- model_version likhne ka ye tarika shi nhi h ... bcoz kal ko production me model ka koi or version aa gya say 2
#toh manually hmme yha aake code update krna padega ... 
#so, hmme chahiye ki yha hum production k latest version ko load kr paye ... 

#now, abb model ka ek URI build krna h ... 
model_uri = f'models:/{model_name}/{model_version}'    #issi uri ki help se hum ... model ko model registry se fetch krte h ... 
model = mlflow.pyfunc.load_model(model_uri)                    #iss line se aapka code execute hoga ... 




#step 03 : bow
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))




#yha se main cheej shuru hoti h ... hum apni website k route create kr skte h ... 
@app.route('/')
def home():
    #return "hello world"
    return render_template('index.html')    #index.html file ko load kr dega and index.html file me hmari website h ... 




@app.route('/predict', methods=['POST'])    #also mujhe yha bta dena h ki ... yha jo data aa rha h ... vo POST me aa rha h
def predict():
    
    #i want ki jo m text daal rha hu ... vo mujhe predict wale page p b chahiye 
    #so, iske liye aapko flask ka request module chahiye ... 
    text = request.form['text']            #and aap yha p bta doge ki kis naam se request bhej rhe h ...    
                                           #index.html me humne text area ka name text daala tha ...     

    #return "working"                        #sirf check krne k liye hum working return kr rhe h
    #return text 



    #prediction ************************************
    #step 01 : load model from the model registry 
    #now, model ko load aap kha kroge? predict() fn k ander ya bhar? 
    #ans is bhar krenge ye kaam ... 
    #kyuki ye jo fn h predict ... ye kitni baar call hoga ? agar hum proper project bna rhe h kisi bdi company 
    #k liye ... toh can u imagine ... ye predict() fn din me kitni baar call hoga ... jitni baar b koi user aaya and usne prediction kiya ... 
    #utni baar predict hoga ... say 1 crore times daily ... 
    #so, do u really think ki 1 crore baar click ho rha h toh 1 crore times mlflow registry se model ko load kre ... 
    #it is not a good strategy ... so hum model registry se model ko upper load krenge ... 
    #see above for this step 
    


    
    #step 02 : cleaning (user se jo text lenge ... usse clean krenge)
    #go to --> flask-aap --> create file --> preprocessing_utility.py 
    normalize_text(text)    #now, preprocessing_utility.py wali file k normalize_text() fn k pass text jayega ... 
                            #and thn text normalize hoga and step 05 me text return ho jayega                    




    #step 03 : bow (so hmme jo text mill rha h ... usse features generate krni h)
    #humne ek gadbad ki h and that is ... hum yha p bow nhi lga skte ... kyuki bow jbb pure 
    #data p train hua tha ... toh humne usse khi save hi nhi kiya tha ... uss vectorizer ko humne save hi nhi kiya tha ...
    #toh abhi mere pass koi vectorizer hi nhi h ... other option is m yha p alag se bow model ko train kru ...
    #which is not a good idea bcoz jo training me hona chahiye vhi prediction me hona chahiye ... 
    #so, humne plan nhi kiya iss cheej ko ... 
    #solution : feature store ... hmme ussi time pe feature store krni thi and abb hum unn 
    #features ko leke aa jate ... but yha p hmare pass feature store ka b option nhi h ... 
    
    #simple solution :- go back --> apni pipeline me chota sa change kr do ... 
    #feature_engineering.py --> bow jha aap lga rhe ho ... vha p aap ... apne vectorizer ko save kr do ... 
    #and usko aap as an artifact b save kr skte ho ... 
    #add line --> pickle.dump(vectorizer, open('models/vectorizer.pkl', 'wb'))
    #dvc repro
    
    features = vectorizer.transform([text])         #dhyan dena ki isko hmesha list of objects dena hota h ... 
    #now, ye aapko palat kr k aapke features dega ... 




    #step 04 : prediction 
    result = model.predict(features)




    #step 05 : show 
    #return text
    #return str(result[0])
    return render_template('index.html', result=result[0])
    #go to index.html and vha p chota sa code or likh do ... 




app.run(debug=True, port=8000)           #port=8000 b daal skte h ... ye isiliye daalte h taki mlflow ka port and flask ka port same na ho jaye 
                                         #but mlflow toh dagshub pe chal rha h ... toh port wale argument ko hum skip kr rhe h
                                         #abb python flask-aap/app.py run kro and isse mujhe webpage p hello world likha milega ... 

#so abb hum front develop krenge ... front end me user ko ek text box dikhayi dega 
#jha p vo text type krega ... and neeche phir predict pe click krna p prediction hoga ... 
#toh ye frontend hum create krenge ... 

#toh front end create krne k liye flask ka funda kya hota h ki ... aapko jitni b html file
#bnani hoti h ... vo aap ek folder k ander bnate ho ... jisko hum templates bulate h ... 

#flask-aap --> templates 
#so, iss templates folder k ander we are going to create a new file --> index.html 
#aapko basic level html aana chahiye ... vo enough hoga ... 