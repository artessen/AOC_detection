# XGBoost Model for Multi-target Classification of Areas of Concern (AOC) for Agriculture in Europe
Pipeline for the implementation of an expert-driven and explainable artificial intelligence model for probabilistically detecting multiple agriculture-related hazards in Europe. 

## Overview
This repository implements an eXtreme Gradient Boosting (XGBoost) model for multi-target classification using a variety of meteorological and environmental features. The implementation leverages R and includes steps for data preprocessing, model training, hyperparameter tuning, and performance evaluation.

The script is designed to handle multiple target variables representing different climate hazards relevant for agriculture in Europe, with each target variable trained independently using customised features and evaluation metrics.

## Processing Steps
- **Data Preprocessing**: Handles missing data, feature selection, and class balancing.
- **Hyperparameter Tuning**: Selection of hyperparameters for each ensemble member to optimise the evaluation metric scores.
- **Ensemble Training**: Builds an ensemble of models for robust predictions.
- **Performance Metrics**: Computes accuracy, precision, recall, and F1 score for training, validation, and test sets.
- **Explainability**: Extracts SHAP (SHapley Additive exPlanations) values for feature importance analysis.
- **Outputs**: Saves trained models, variable importance, SHAP values, and prediction results.

## Types of Areas of Concern (AOC)
- **ColdS**: Cold Spell
- **HeatW**: Heatwave
- **Drought**: Droughts
- **HotDry**: Hot and Dry conditions
- **RainS**: Rain Surplus
- **RainD**: Rain Deficit
- **TSumS**: Temperature Accumulation Surplus
- **TSumD**: Temperature Accumulation Deficit

The .rds files inside the folders with the AOC acronyms are the pre-trained ensemble members for the detection of AOC in Europe. 

More details regarding the definition of AOC types can be found in the following web addresses:
- https://joint-research-centre.ec.europa.eu/monitoring-agricultural-resources-mars/jrc-mars-bulletin_en
- https://doi.org/10.1016/j.agsy.2018.07.003

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/artessen/AOC_detection.git
   ```
2. Install the required R libraries:
   ```R
   install.packages(c("data.table", "caret", "xgboost", "Metrics", "ggplot2", "parallel", "lubridate", "reshape2", "haven", "dplyr", "matrixStats", "tidyverse"))
   ```

## Usage

### Data Preparation
- Input data is expected in a CSV file (e.g., `dfDatabase_sample.csv` for training, `dfDatabaseProd_sample.csv` for inference).
- The files `dfDatabase_sample.csv` and `dfDatabaseProd_sample.csv` are samples of how the input CSV files should look like.
- The original AOC data used as target variables are available as open-access datasets at https://data.jrc.ec.europa.eu/.
- The script identifies features, target variables, and ID columns dynamically.

### Custom Functions
- Custom objective and evaluation functions are defined in `00.a.aux_AuxiliaryFunctions.R`.
- Make sure that the file `00.a.aux_AuxiliaryFunctions.R` is available in the working directory.

### Running the Script
1. Modify the `dfData` variable to point to your custom input data file for training purposes.
2. For training of a new model, run the script:
   ```R
   Rscript 00.a.fTrainModel.R
   ```
3. For inference of a trained model, run the script:
   ```R
   Rscript 00.b.fPredictModel.R
   ```

### Outputs
For each target variable, the following files are generated:
- **Trained Models**: Saved as RDS files in target-specific folders.
- **Predictions**: CSV files containing mean and standard deviation of predictions.
- **Performance Metrics**: CSV files with accuracy, recall, precision, and F1 scores.
- **Variable Importance**: CSV files for feature importance metrics (Gain, Cover, Frequency, SHAP).

## Implementation Details
1. **Input Data**: 
   - Features are divided into mean, anomaly, and specific climate indicators.
   - Target variables include climate hazards such as drought, heatwaves, and heavy rainfall.

2. **Model Training**:
   - Data is split into training, validation, and test sets.
   - Classes are balanced using a custom `scale_pos_weight` parameter.
   - Models are trained using `xgb.train` with early stopping to prevent overfitting.

3. **Ensemble Creation**:
   - 1000 ensemble members are trained with randomised hyperparameters.
   - Outputs include ensemble predictions and error metrics.

4. **Explainability**:
   - SHAP values are calculated to provide feature importance.
   - Results are visualised using beeswarm plots.

## Example Plots
- Performance metrics per target variable.
- SHAP summary plot for feature contributions.

## License
This project is licensed under the GNU General Public License v3.0. See the LICENSE file for details.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any feature suggestions or bug fixes.

## Acknowledgements
This work utilises open-source libraries and builds upon the extensive functionality provided by the XGBoost library (https://CRAN.R-project.org/package=xgboost). Special thanks to the contributors of the R packages used in this project.

## Citation
Essenfelder, A. H., Toreti, A., Seguini, L. "Expert-driven explainable artificial intelligence models can detect multiple climate hazards relevant for agriculture"
