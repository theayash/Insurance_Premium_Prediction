From continuumio/anaconda3:4.4.0

ADD . /app
WORKDIR /app
RUN pip install -r requirements.txt