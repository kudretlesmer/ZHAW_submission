FROM python:3.12.4-slim
 
WORKDIR /code

RUN pip install pipenv

COPY ./Pipfile /code/Pipfile
COPY ./Pipfile.lock /code/Pipfile.lock

RUN pipenv install
 
COPY ./app /code/app

CMD ["pipenv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]