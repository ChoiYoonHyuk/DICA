"""Run the final complete-decoder experiment.

Example on the experiment server:
    python main.py --family qwen3 --output NEW_RESULTS
Use --help for model, document and output path options.
"""
from host_generation_benchmark import main

if __name__ == "__main__":
    main()
