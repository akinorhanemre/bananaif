FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /

COPY . .

# Install git and g++
RUN apt-get update && apt-get install -y git g++

RUN pip install -r requirements.txt

# Install xformers directly
RUN pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
RUN pip install deepfloyd-if==1.0.1

EXPOSE 8000

CMD ["python", "app.py"]
