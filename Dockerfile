FROM python:3.8-slim

RUN pip install --upgrade pip
RUN apt-get update && \
      apt-get install libgomp1
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY bike_sharing_demand /bs/bike_sharing_demand

WORKDIR /bs

CMD ["python", "-m", "bike_sharing_demand"]
