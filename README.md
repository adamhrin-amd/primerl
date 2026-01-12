## Request GPU resources

```sh
srun -p amd-tw-verification -N 1 -G 8 -t 08:00:00 --pty bash
```

## Load container launcher and start the vllm Open-AI compatible server

```sh
./interactive_vllm.sh
```

## Setup prime and run vf-eval

```sh
# install uv if not installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# install prime if not installed
uv tool install prime

# init venv
uv venv --python 3.12
source .venv/bin/activate

# install env package locally in editable mode
prime env pull kalomaze/alphabet-sort
cd alphabet-sort
uv pip install -e .

# run package
export OPENAI_API_KEY="dummy"
uv run vf-eval alphabet-sort \
  -m /shared_silo/scratch/models/DeepSeek-V3 \
  --api-base-url http://localhost:8000/v1
```