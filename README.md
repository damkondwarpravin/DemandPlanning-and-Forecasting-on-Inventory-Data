# Demand Planning and Forecasting on Inventory Data

This project is a complete assignment folder for demand forecasting and inventory planning using a large FMCG sales dataset.

## Objective
Build a demand planning workflow that:
- analyzes sales and inventory-related features,
- prepares daily demand data,
- trains a predictive model for future demand,
- evaluates forecast accuracy,
- estimates safety stock and reorder point for inventory planning.

## Dataset
The project uses a 3-year FMCG sales dataset with around 1.1M rows located at:
- `data/archive/fmcg_sales_3years_1M_rows.csv`

## Project structure
- `notebooks/DemandplanningAndForecasting.ipynb` – complete analysis notebook
- `src/demand_forecasting.py` – reusable Python script for forecasting and inventory policy
- `src/rebuild_notebook.py` – notebook rebuild helper
- `data/` – dataset and generated outputs
- `requirements.txt` – required Python packages

## Workflow summary
1. Load sales data and convert date fields
2. Aggregate daily demand and operational features
3. Create lag and rolling features for forecasting
4. Train a Random Forest Regressor
5. Measure MAE, RMSE, and R²
6. Estimate safety stock and reorder point
7. Save forecast results and visualizations

## How to run
From the project root:

```bash
python src/demand_forecasting.py
```

Or open the notebook in Jupyter/VS Code and run the cells.

## Tech stack
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Lecture / training reference
Add your lecture or class link here:
- Lecture URL: https://example.com/your-lecture-link

## GitHub upload
This folder is structured and ready to be uploaded as a single GitHub repository.

Steps:
1. Create a new GitHub repository
2. Upload this folder contents
3. Add the repository URL in your assignment submission
4. Optionally add screenshots or a short summary in the repository description

## Notes
This project is designed to be a clean, shareable assignment and can be submitted as a GitHub portfolio project.
