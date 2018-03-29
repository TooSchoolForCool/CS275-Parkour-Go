PHONY: train demo-simple demo-hardcore

train:
	python3 ./src/main.py --mode=train --env="BipedalWalker-v2" --n_workers=4 \
		--n_frames=1 --n_steps=20 --model_load_dir="./models/CS275-BipedalWalker-v2" \
		--model_save_dir="./models/new-BipedalWalker-v2"

demo-simple:
	python3 ./src/main.py --mode=test --env="BipedalWalker-v2" --n_steps=20 \
		--n_frames=1 --model_load_dir="./models/BipedalWalker-v2"

demo-hardcore:
	python3 ./src/main.py --mode=test --env="BipedalWalkerHardcore-v2" --n_steps=100 \
		--networks=CONV --n_frames=4 --model_load_dir="./models/86hr-BipedalWalkerHardcore-v2"