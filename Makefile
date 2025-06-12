#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = sa
PYTHON_INTERPRETER = python

NUM_TRAIN_BATCHES = 1000
NUM_TEST_BATCHES = 10
NUM_TRAINING_EPOCHS = 100
NUM_TRAIN_SET_IMAGES_TO_VISUALISE = 200
NUM_TEST_SET_IMAGES_TO_VISUALISE = 20

ifeq ($(OS),Windows_NT)
    detected_OS := Windows

    export PYTHON_INTERPRETER = python
    export PYTHON_VERSION = 3.11
    export CUDA_VISIBLE_DEVICE = "0"
	export BACKEND = jax

    export SET_CMD = set
    export AND_CMD = &
else
	PYTHON_VERSION = 3.10
    detected_OS := $(shell sh -c 'uname 2>/dev/null || echo Unknown')

    PYTHON_INTERPRETER = python
    CUDA_VISIBLE_DEVICE = "0"
	BACKEND =jax

    SET_CMD =
    AND_CMD =

    PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
	BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')

    ifeq (,$(shell which conda))
		HAS_CONDA=False
	else
		HAS_CONDA=True
	endif
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format





## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"




#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

# JAX_DEBUG_NANS=True

test_fil:
	CUDA_VISIBLE_DEVICES="1" KERAS_BACKEND="jax" PYTHONPATH=. $(PYTHON_INTERPRETER) fil/fi_test.py
kuhn:
	CUDA_VISIBLE_DEVICES="1" KERAS_BACKEND="jax" PYTHONPATH=. $(PYTHON_INTERPRETER) deepcfr/main_deep_cfr_keras_jax.py --config=../configs/kuhn.py
kuhn_torch:
	PYTHONPATH=. $(PYTHON_INTERPRETER) deepcfr/main_deep_cfr_pytorch.py --config=../configs/kuhn.py

leduc:
	PYTHONPATH=. $(PYTHON_INTERPRETER) deepcfr/main_deep_cfr_pytorch.py --config=../configs/leduc.py

install_server_torch:
	uv pip install torch torchvision torchaudio

install_server_jax:
	uv pip install -U "jax[cuda12]"


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
