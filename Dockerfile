FROM python:3.9.9

COPY ./ ./

RUN python3 -m pip install -r ./requirements.txt 

CMD gunicorn -c va/service/vaGunicornConfig.py service.vaApp:app & python3 tool/api_service.py


     