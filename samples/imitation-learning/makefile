# define python runtime
PYTHON = python3

# define OpenAI MuJoCo Environment
GYM_ENV = Humanoid-v2
# GYM_ENV = Walker2d-v2

# define expert file path
DATASET_PATH = data/$(GYM_ENV)-data.pkl


.PHONY: run


run:
	$(PYTHON) main.py --expert $(DATASET_PATH) --env $(GYM_ENV)
