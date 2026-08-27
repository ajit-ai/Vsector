.PHONY: install test lint run docker proto

install:
	pip install -e .[dev]
proto:
	python -m grpc_tools.protoc -I proto --python_out=vsector/api --grpc_python_out=vsector/api proto/vsector.proto
test:
	pytest -q
lint:
	ruff check vsector
run:
	vsector serve --reload
docker:
	docker build -t vsector:0.1.0 .
	@echo "docker-compose up --build"
