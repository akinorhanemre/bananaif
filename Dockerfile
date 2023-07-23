FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /

COPY . .

# Install git
RUN apt-get update && apt-get install -y git

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["python", "app.py"]