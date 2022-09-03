#!/bin/bash
find . -name "*py"|xargs python3 -m autoflake -i --remove-all-unused-imports --remove-unused-variables --ignore-init-module-imports
python3 -m isort .
python3 -m black .
python3 -m flake8 .
