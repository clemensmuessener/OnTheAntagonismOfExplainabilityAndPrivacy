# XAIandDataPrivacy

Run command `docker build -t xaidataprivacy .` in the directory with the Dockerfile.

Then you can run the image mounted to your working directory with the following command: `docker run -p 8888:8888 -v WORKING_DIRECTORY:/home/jovyan/work xaidataprivacy` or `docker run -p 8888:8888 -v "$(pwd):/home/jovyan/work" xaidataprivacy`