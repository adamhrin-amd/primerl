## Request GPU resources

```sh
srun -p amd-tw-verification -N 1 -G 8 -t 08:00:00 --pty bash
```

## Load container launcher and start the vllm Open-AI compatible server

```sh
./interactive_vllm.sh
```

## Run vf-eval

```sh
uv run vf-eval -m /shared_silo/scratch/models/DeepSeek-V3 --api-base-url http://localhost:8000/v1
```