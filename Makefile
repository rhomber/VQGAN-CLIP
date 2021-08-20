all: build

build:
	docker build -t davidradunz/vqgan-clip .

run:
	docker run --runtime=nvidia -i -t --rm davidradunz/vqgan-clip:latest
