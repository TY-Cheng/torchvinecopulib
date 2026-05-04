# Thin local task runner that mirrors the existing uv + GitHub Actions workflow.
# Main flow:
#   just setup
#   just test
#   just examples
#   just docs
#   just bench
#   just workflow

set dotenv-load := true

default:
    @just --list

[private]
_require-external-uv-env:
    @test -n "${UV_PROJECT_ENVIRONMENT:-}" || { \
      printf '%s\n' 'UV_PROJECT_ENVIRONMENT is not set.'; \
      printf '%s\n' 'Set it in .env so uv uses the shared external environment instead of a project-local .venv.'; \
      exit 1; \
    }
    @case "${UV_PROJECT_ENVIRONMENT}" in \
      /*) ;; \
      *) \
        printf '%s\n' 'UV_PROJECT_ENVIRONMENT must be an absolute path.'; \
        printf '%s\n' "Current value: ${UV_PROJECT_ENVIRONMENT}"; \
        exit 1; \
        ;; \
    esac
    @repo_root="$(pwd -P)"; \
    env_path="$(python3 -c "from pathlib import Path; import os; print(Path(os.environ['UV_PROJECT_ENVIRONMENT']).resolve(strict=False))")"; \
    case "$env_path" in \
      "$repo_root"| "$repo_root"/*) \
        printf '%s\n' 'UV_PROJECT_ENVIRONMENT must point outside the repository.'; \
        printf '%s\n' "Resolved value: $env_path"; \
        printf '%s\n' "Repository root: $repo_root"; \
        exit 1; \
        ;; \
    esac

env: _require-external-uv-env
    @printf 'UV_PROJECT_ENVIRONMENT=%s\n' "${UV_PROJECT_ENVIRONMENT:-<unset>}"
    uv run --extra cpu python -c "import sys; import torch; print(sys.executable); print(torch.cuda.is_available())"

setup: _require-external-uv-env
    uv sync --extra cpu --extra reference --group docs --group examples

test mode="auto": _require-external-uv-env
    @case "{{ mode }}" in \
      auto|default) \
        CUDA_FLAG="$(uv run --extra cpu --extra reference python -c 'import torch; print(1 if torch.cuda.is_available() else 0)')"; \
        if [ "$CUDA_FLAG" = "1" ]; then \
          TVC_RUN_CUDA_TESTS=1 \
          TVC_HYPOTHESIS_MAX_EXAMPLES="${TVC_HYPOTHESIS_MAX_EXAMPLES:-100}" \
          uv run --extra cpu --extra reference pytest \
            --cov=torchvinecopulib \
            --cov-branch \
            --cov-report=term-missing \
            --cov-report=xml:coverage.xml \
            --cov-report=html \
            --cov-fail-under=94 \
            -W error::DeprecationWarning \
            tests; \
        else \
          TVC_HYPOTHESIS_MAX_EXAMPLES="${TVC_HYPOTHESIS_MAX_EXAMPLES:-100}" \
          uv run --extra cpu --extra reference pytest \
            --cov=torchvinecopulib \
            --cov-branch \
            --cov-report=term-missing \
            --cov-report=xml:coverage.xml \
            --cov-report=html \
            --cov-fail-under=94 \
            -W error::DeprecationWarning \
            -m "not cuda" \
            tests; \
        fi; \
        ;; \
      cpu) \
        TVC_HYPOTHESIS_MAX_EXAMPLES="${TVC_HYPOTHESIS_MAX_EXAMPLES:-100}" \
        uv run --extra cpu --extra reference pytest \
          --cov=torchvinecopulib \
          --cov-branch \
          --cov-report=term-missing \
          --cov-report=xml:coverage.xml \
          --cov-report=html \
          --cov-fail-under=94 \
          -W error::DeprecationWarning \
          -m "not cuda" \
          tests; \
        ;; \
      *) \
        printf '%s\n' "unknown test mode: {{ mode }}"; \
        printf '%s\n' "use: just test      # installs reference and auto-runs CUDA tests when available"; \
        printf '%s\n' "     just test cpu  # force the CPU-only suite"; \
        exit 1; \
        ;; \
    esac

examples mode="write": _require-external-uv-env
    @case "{{ mode }}" in \
      write) \
        uv run --extra cpu --group examples python scripts/generate_example_assets.py; \
        uv run ruff format .; \
        ;; \
      check) \
        uv run --extra cpu --group examples python scripts/generate_example_assets.py --check; \
        ;; \
      *) \
        printf '%s\n' "unknown examples mode: {{ mode }}"; \
        printf '%s\n' "use: just examples        # regenerate assets"; \
        printf '%s\n' "     just examples check  # verify committed assets"; \
        exit 1; \
        ;; \
    esac

docs: _require-external-uv-env
    uv run --extra cpu --group docs mkdocs build --strict
    uv run --extra cpu pytest tests/test_docs_smoke.py -q

bench: _require-external-uv-env
    uv run --extra cpu python benchmarks/profile_builder.py \
      --num-obs 256 \
      --num-dim 4 \
      --grid-size 33 \
      --output benchmarks/results/profile_builder.ci.json
    uv run --extra cpu python benchmarks/profile_query.py \
      --num-obs 256 \
      --num-dim 4 \
      --grid-size 33 \
      --batch-size 32 \
      --cdf-samples 127 \
      --compile-backend eager \
      --output benchmarks/results/profile_query.ci.json

workflow: _require-external-uv-env
    just setup
    just test
    just examples
    just docs
