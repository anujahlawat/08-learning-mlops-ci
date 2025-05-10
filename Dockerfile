#FROM python:3.10-slim
FROM python:3.9

WORKDIR /app
#docker container m /app directory build kr rha hu m 

COPY flask_app/ /app/
#flask app ko copy kr rha hu dcoker container ki /app directory m 

COPY models/vectorizer.pkl /app/models/vectorizer.pkl
#models/vectorizer.pkl ... ko m copy kr rha hu ... app k ander models bol k folder 
#bnega and uske ander ... vectorizer.pkl paste ho jayega ... 

RUN pip install -r requirements.txt
#see above hum flask_app folder ko hi copy kr rhe h ... and uski requirements.txt file 
#execute hogi ... 

RUN python -m nltk.downloader stopwords wordnet
#add this statement bcoz execution p stopwords downlaod nhi huae ... toh abb ho jayenge ... 

EXPOSE 5000

#CMD ["python","app.py"]
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]