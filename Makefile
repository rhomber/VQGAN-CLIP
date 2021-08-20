all: build

build:
	docker build -t davidradunz/vqgan-clip .

run:
	docker run --runtime=nvidia -i -t --rm davidradunz/vqgan-clip:latest

push:
	docker push davidradunz/vqgan-clip:latest

prune:
	docker container prune
