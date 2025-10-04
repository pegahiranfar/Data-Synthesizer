# * An example*

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src import SimpleTimeSeriesGenerator, SimpleTabularGenerator

# 1. Time series data
# Create a time series generator
ts_gen = SimpleTimeSeriesGenerator(seed=42)

# Generate 500 data with upward trend and daily patterns
ts_data = ts_gen.generate(n_points=500,trend='up', seasonality=True, noise_level=0.2)

print(ts_data.head())
ts_gen.save_csv(ts_data, 'output_timeseries.csv')


# 2. Classification data -- easy
# Create a tabular data generator
tab_gen = SimpleTabularGenerator(seed=42)

# Generate classification dataset
clf_gen = tab_gen.generate_classification( n_samples=500, n_features=5, n_classes=2, difficulty='easy')

tab_gen.show_info(clf_gen)
tab_gen.save_csv(clf_gen, 'output_clf_easy.csv')


# 3. Classification data -- Hard + Imbalanced
clf_hard = tab_gen.generate_classification(n_samples=1000, n_features=10, n_classes=2, difficulty='hard')

# Make it imbalanced 
clf_imbalanced = tab_gen.make_imbalanced(clf_hard, minority_ratio=0.05)

tab_gen.show_info(clf_imbalanced)
tab_gen.save_csv(clf_imbalanced, 'output_clf_imbalanced.csv')


# 4. Multi-class classification
# Generate dataset with 5 classes
clf_multi = tab_gen.generate_classification(n_samples=1000, n_features=8, n_classes=5, difficulty='medium')

tab_gen.show_info(clf_multi)
tab_gen.save_csv(clf_multi, 'output_clf_multi.csv')


# 5. Regression Data
# Generate regression dataset
reg_data = tab_gen.generate_regression(n_samples=800, n_features=6, noise_level=5.0)

tab_gen.show_info(reg_data)
tab_gen.save_csv(reg_data, 'output_reg.csv')
