# Dataset Analysis

This project analyzes Absa admission data to investigate the performance of a machine learning model and any potential discrimination in the admission process.

## Data

The data used in this project comes from two CSV files located in the `data/` directory:

1. `data/candidates.csv` - This file contains information about the candidates, including their age, gender, number of children, and employee card ID.

2. `data/admissions.csv` - This file contains information about the admissions process, including the candidate ID, department, and admit status.

## Project Structure

The project consists of two main Python files:

1. `data_processing.py` - Contains functions for data loading, preprocessing, feature selection, model training, and model evaluation.

2. `main.py` - The main script that calls the functions from `data_processing.py` to perform the analysis.

## How to Run

1. Install the required Python packages using pip:
pip install -r requirements.txt

2. Run the `main.py` script:
python main.py
## Results

The results suggest that there is discrimination based on gender. The admission rate for males is significantly higher than for females, and the chi-square statistic is very high, indicating that the difference between the observed and expected distributions of admission rates for males and females is statistically significant.

## License

This project is released under the [GNU General Public License v3.0](LICENSE).
