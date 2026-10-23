# Build, container, and test entry points.
#
#   make wheel   -> dist/*.whl + dist/requirements.lock (hash-locked)
#   make image   -> container image from the wheel, using podman or docker
#   make run     -> run the image locally with credentials from your shell
#   make test    -> run the test suite

IMAGE ?= langchain-harness:0.1.0
CONTAINER_ENGINE ?= $(shell command -v podman >/dev/null 2>&1 && echo podman || echo docker)
# Podman defaults to OCI format, which drops HEALTHCHECK. Docker rejects the flag.
CONTAINER_FORMAT ?= $(if $(filter podman,$(CONTAINER_ENGINE)),--format docker,)
PROFILE_DIR ?= deploy/agent

.PHONY: wheel test image run clean

wheel:
	rm -rf dist
	uv build --wheel --out-dir dist
	uv export --no-dev --no-emit-project --extra postgres --format requirements-txt \
		--output-file dist/requirements.lock
	@echo "Built $$(ls dist/*.whl)"

test:
	uv run --extra dev pytest -q

image: wheel
	$(CONTAINER_ENGINE) build $(CONTAINER_FORMAT) \
		--build-arg PROFILE_DIR=$(PROFILE_DIR) \
		-t $(IMAGE) .

run: image
	$(CONTAINER_ENGINE) run --rm -p 8000:8000 \
		-e HARNESS_PROVIDER \
		-e HARNESS_MODEL \
		-e HARNESS_API_KEY \
		$(IMAGE)

clean:
	rm -rf dist
