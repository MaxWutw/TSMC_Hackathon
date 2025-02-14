FROM python:3.9.9

COPY ./ ./

RUN python3 -m pip install -r ./requirements.txt 
    


     