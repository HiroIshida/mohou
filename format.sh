#!/bin/bash
find . -name "*py"|xargs autoflake -i --remove-all-unused-imports --remove-unused-variables --ignore-init-module-imports 
isort . 
black --required-version 22.3.0 .
flake8 .
