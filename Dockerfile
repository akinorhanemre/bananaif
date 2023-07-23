WORKDIR /

COPY . .

RUN apt-get update && apt-get install -y git g++

RUN pip install -r requirements.txt

RUN pip install xformers==0.0.16 > /dev/null
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu117 torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1
RUN pip install git+https://github.com/openai/CLIP.git

EXPOSE 8000

CMD ["python", "app.py"]
