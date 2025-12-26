# Reproducing the Analyses

1. **Installation**: `pip install -r code/requirements.txt`
2. **Data collection**: Run the scripts in order:
```bash
cd code/01_data_collection
python manual_performance_dataset.py
python collect_data.py

cd ../02_data_processing
python merge_data.py
python clean_data.py

cd ../03_analysis
python meta_analysis.py
cd ../04_visualization
python generate_all_figures.py

