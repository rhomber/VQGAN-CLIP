all: build

build:
	docker build -t rhomber/vqgan-clip .

run:
	docker run --runtime=nvidia -i -t --rm rhomber/vqgan-clip:latest

push:
	docker push rhomber/vqgan-clip:latest

prune:
	docker container prune
