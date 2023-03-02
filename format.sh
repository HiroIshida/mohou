#!/bin/bash
find . -name "*py"|xargs python3 -m autoflake -i --remove-all-unused-imports --remove-unused-variables --ignore-init-module-imports
python3 -m isort .
python3 -m black --required-version 22.6.0 .
python3 -m flake8 .
