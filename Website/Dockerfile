FROM python:3

ENV PYTHONUNBUFFERED 1
RUN mkdir -p /opt/services/flaskapp/src
#VOLUME ["/opt/services/flaskapp/src"]
# We copy the requirements.txt file first to avoid cache invalidations
COPY ./Website/requirements.txt /opt/services/flaskapp/src/
WORKDIR /opt/services/flaskapp/src
RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip install -r requirements.txt
COPY ./Website/* /opt/services/flaskapp/src
# RUN rm /opt/services/flaskapp/src/Text_Gen_Model 
COPY ./Text_Gen_Model /opt/services/flaskapp/src/Text_Gen_Model
WORKDIR /opt/services/flaskapp/src/
EXPOSE 5080
CMD ["python", "app.py"]
