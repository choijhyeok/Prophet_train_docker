FROM python:3.7-stretch

WORKDIR /usr/src/app


RUN apt-get -y install libc-dev

RUN pip install pip==19.1.1

COPY requirements.txt ./
RUN pip install -r requirements.txt
RUN pip install ipython==7.5.0

RUN pip3 install --no-cache boto3 botocore joblib flask logger scikit-learn seaborn
RUN pip install fbprophet



RUN mkdir -p /usr/src/app/tmp_files
RUN chmod 755 /usr/src/app/tmp_files

COPY bootstrap.sh ./
COPY train_ ./train_

RUN chmod 755 /usr/src/app/bootstrap.sh
RUN chmod +x /usr/src/app/bootstrap.sh


EXPOSE 50021
ENTRYPOINT ["/usr/src/app/bootstrap.sh"]

