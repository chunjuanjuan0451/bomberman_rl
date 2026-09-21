#!/usr/bin/env bash
set -euo pipefail

# Exercise the repository Dockerfile with tournament-like limits.
# On Apple Silicon, use linux/arm64 for fast functional checks and linux/amd64
# only for compatibility checks; emulation latency is not a Ryzen benchmark.
platform="${BOMBERMAN_DOCKER_PLATFORM:-linux/arm64}"
image="${BOMBERMAN_DOCKER_IMAGE:-bomberman-rl-official:${platform##*/}}"
dockerfile="${BOMBERMAN_DOCKERFILE:-Dockerfile}"
skip_build="${BOMBERMAN_DOCKER_SKIP_BUILD:-0}"

if [[ "${skip_build}" != "1" ]]; then
  docker build --platform "${platform}" --progress=plain \
    -f "${dockerfile}" -t "${image}" .
fi

echo "dockerfile=${dockerfile} platform=${platform} image=${image}"

run_limited() {
  docker run --rm \
    --platform "${platform}" \
    --cpus 1 \
    --memory 8g \
    --pids-limit 512 \
    -e OMP_NUM_THREADS=1 \
    -e MKL_NUM_THREADS=1 \
    -e OPENBLAS_NUM_THREADS=1 \
    -e NUMEXPR_NUM_THREADS=1 \
    -e SDL_AUDIODRIVER=dummy \
    -e SDL_VIDEODRIVER=dummy \
    -e XDG_RUNTIME_DIR=/tmp \
    "${image}" "$@"
}

echo "[1/3] Runtime, PyTorch, cgroup limits, and frozen v4 checkpoint"
run_limited python tools/check_runtime_environment.py \
  --require-container-limits --load-v4-checkpoint

echo "[2/3] Official main.py smoke with the PyTorch v4 checkpoint"
run_limited python main.py play \
  --agents model_a_dqn random_agent random_agent random_agent \
  --train 0 --continue-without-training --scenario coin-heaven \
  --no-gui --n-rounds 1 --seed 9901

echo "[3/3] Official main.py smoke with the final niulai agent"
run_limited python main.py play \
  --agents niulai random_agent random_agent random_agent \
  --train 0 --continue-without-training --scenario classic \
  --no-gui --n-rounds 1 --seed 9902

docker image inspect "${image}" \
  --format 'image={{.RepoTags}} id={{.Id}} architecture={{.Architecture}}'
