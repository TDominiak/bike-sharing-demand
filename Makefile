SVC_NAME = bike-sharing-demand
RELEASE=latest

ifdef RELEASE
	VERSION = $(RELEASE)
else
	VERSION = `git rev-parse --short HEAD`
endif

DOCKER_IMAGE = $(SVC_NAME):$(VERSION)

## run: runs app locally
run:
	python -m bike_sharing_demand

## push: pushes service image to docker registry

docker-run: build-docker
	docker run -v $(MOUNT):/bs/data -it $(DOCKER_IMAGE)

## build: build docker image
build-docker:
	docker build -t $(DOCKER_IMAGE) .
