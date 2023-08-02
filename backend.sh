# python -u backend.py --config_file config/develop.yaml --output_file test.json --log_file test.log 
export CONFIG_FILE=config/develop.yaml
export OUTPUT_FILE=test.json
export LOG_FILE=test.log
uvicorn backend:app --host 0.0.0.0 --port 18888 