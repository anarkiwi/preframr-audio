#!/bin/sh
set -e
black --check preframr_audio tests
pylint preframr_audio tests
pyright preframr_audio
pytest --cov=preframr_audio --cov-report=term-missing --cov-fail-under=85
