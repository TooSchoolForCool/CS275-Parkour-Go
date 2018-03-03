# define python runtime
PYTHON = python3

# define OpenAI MuJoCo Environment
GYM_ENV = Humanoid-v2
# GYM_ENV = Walker2d-v2
# GYM_ENV = Hopper-v2

# define expert file path
DATASET_PATH = data/$(GYM_ENV)-data.pkl


.PHONY: run getdata


run:
	$(PYTHON) main.py --expert=$(DATASET_PATH) --env=$(GYM_ENV) --rollout=10

getdata:
	$(PYTHON) run_expert.py experts/$(GYM_ENV).pkl $(GYM_ENV) --num_rollouts=20
