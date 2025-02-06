import numpy as np
import pandas as pd
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import os

# Load the dataset from the specified directory
file_path = r"E:\Piezoelectric\data\NFIS_training_data.csv"
dataset = pd.read_csv(file_path)

# Define input variables
voltage = dataset['Voltage (V)']
temperature = dataset['Temperature (°C)']
humidity = dataset['Humidity (%RH)']
output_adjustment = dataset['Output Adjustment (N)']

# Step 1: Fuzzification of inputs using Gaussian membership functions
# Create fuzzy control variables
voltage_ctrl = ctrl.Antecedent(np.arange(0.1, 5.1, 0.1), 'Voltage')
temperature_ctrl = ctrl.Antecedent(np.arange(10, 41, 0.5), 'Temperature')
humidity_ctrl = ctrl.Antecedent(np.arange(20, 91, 1), 'Humidity')
output_ctrl = ctrl.Consequent(np.arange(-1, 1.1, 0.1), 'Output Adjustment')

# Define Gaussian membership functions for inputs
voltage_ctrl['Low'] = fuzz.gaussmf(voltage_ctrl.universe, 0.1, 0.5)
voltage_ctrl['Medium'] = fuzz.gaussmf(voltage_ctrl.universe, 2.5, 1.0)
voltage_ctrl['High'] = fuzz.gaussmf(voltage_ctrl.universe, 4.5, 0.5)

temperature_ctrl['Low'] = fuzz.gaussmf(temperature_ctrl.universe, 10, 3)
temperature_ctrl['Medium'] = fuzz.gaussmf(temperature_ctrl.universe, 25, 5)
temperature_ctrl['High'] = fuzz.gaussmf(temperature_ctrl.universe, 40, 3)

humidity_ctrl['Low'] = fuzz.gaussmf(humidity_ctrl.universe, 20, 10)
humidity_ctrl['Medium'] = fuzz.gaussmf(humidity_ctrl.universe, 50, 15)
humidity_ctrl['High'] = fuzz.gaussmf(humidity_ctrl.universe, 90, 10)

# Define membership functions for the output
output_ctrl['Decrease'] = fuzz.gaussmf(output_ctrl.universe, -0.5, 0.2)
output_ctrl['No Change'] = fuzz.gaussmf(output_ctrl.universe, 0.0, 0.1)
output_ctrl['Increase'] = fuzz.gaussmf(output_ctrl.universe, 0.5, 0.2)

# Step 2: Define fuzzy rules for tuning
# Step 2: Define fuzzy rules for tuning
rule1 = ctrl.Rule(voltage_ctrl['Low'] & temperature_ctrl['Low'] & humidity_ctrl['Low'], output_ctrl['Decrease'])
rule2 = ctrl.Rule(voltage_ctrl['Low'] & temperature_ctrl['Low'] & humidity_ctrl['Medium'], output_ctrl['Decrease'])
rule3 = ctrl.Rule(voltage_ctrl['Low'] & temperature_ctrl['Low'] & humidity_ctrl['High'], output_ctrl['No Change'])

rule4 = ctrl.Rule(voltage_ctrl['Low'] & temperature_ctrl['Medium'] & humidity_ctrl['Low'], output_ctrl['Decrease'])
rule5 = ctrl.Rule(voltage_ctrl['Low'] & temperature_ctrl['Medium'] & humidity_ctrl['Medium'], output_ctrl['No Change'])
rule6 = ctrl.Rule(voltage_ctrl['Low'] & temperature_ctrl['Medium'] & humidity_ctrl['High'], output_ctrl['Increase'])

rule7 = ctrl.Rule(voltage_ctrl['Low'] & temperature_ctrl['High'] & humidity_ctrl['Low'], output_ctrl['No Change'])
rule8 = ctrl.Rule(voltage_ctrl['Low'] & temperature_ctrl['High'] & humidity_ctrl['Medium'], output_ctrl['Increase'])
rule9 = ctrl.Rule(voltage_ctrl['Low'] & temperature_ctrl['High'] & humidity_ctrl['High'], output_ctrl['Increase'])

rule10 = ctrl.Rule(voltage_ctrl['Medium'] & temperature_ctrl['Low'] & humidity_ctrl['Low'], output_ctrl['Decrease'])
rule11 = ctrl.Rule(voltage_ctrl['Medium'] & temperature_ctrl['Low'] & humidity_ctrl['Medium'], output_ctrl['No Change'])
rule12 = ctrl.Rule(voltage_ctrl['Medium'] & temperature_ctrl['Low'] & humidity_ctrl['High'], output_ctrl['Increase'])

rule13 = ctrl.Rule(voltage_ctrl['Medium'] & temperature_ctrl['Medium'] & humidity_ctrl['Low'], output_ctrl['Decrease'])
rule14 = ctrl.Rule(voltage_ctrl['Medium'] & temperature_ctrl['Medium'] & humidity_ctrl['Medium'], output_ctrl['No Change'])
rule15 = ctrl.Rule(voltage_ctrl['Medium'] & temperature_ctrl['Medium'] & humidity_ctrl['High'], output_ctrl['Increase'])

rule16 = ctrl.Rule(voltage_ctrl['Medium'] & temperature_ctrl['High'] & humidity_ctrl['Low'], output_ctrl['No Change'])
rule17 = ctrl.Rule(voltage_ctrl['Medium'] & temperature_ctrl['High'] & humidity_ctrl['Medium'], output_ctrl['Increase'])
rule18 = ctrl.Rule(voltage_ctrl['Medium'] & temperature_ctrl['High'] & humidity_ctrl['High'], output_ctrl['Increase'])

rule19 = ctrl.Rule(voltage_ctrl['High'] & temperature_ctrl['Low'] & humidity_ctrl['Low'], output_ctrl['No Change'])
rule20 = ctrl.Rule(voltage_ctrl['High'] & temperature_ctrl['Low'] & humidity_ctrl['Medium'], output_ctrl['Increase'])
rule21 = ctrl.Rule(voltage_ctrl['High'] & temperature_ctrl['Low'] & humidity_ctrl['High'], output_ctrl['Increase'])

rule22 = ctrl.Rule(voltage_ctrl['High'] & temperature_ctrl['Medium'] & humidity_ctrl['Low'], output_ctrl['No Change'])
rule23 = ctrl.Rule(voltage_ctrl['High'] & temperature_ctrl['Medium'] & humidity_ctrl['Medium'], output_ctrl['Increase'])
rule24 = ctrl.Rule(voltage_ctrl['High'] & temperature_ctrl['Medium'] & humidity_ctrl['High'], output_ctrl['Increase'])

rule25 = ctrl.Rule(voltage_ctrl['High'] & temperature_ctrl['High'] & humidity_ctrl['Low'], output_ctrl['Increase'])
rule26 = ctrl.Rule(voltage_ctrl['High'] & temperature_ctrl['High'] & humidity_ctrl['Medium'], output_ctrl['Increase'])
rule27 = ctrl.Rule(voltage_ctrl['High'] & temperature_ctrl['High'] & humidity_ctrl['High'], output_ctrl['Decrease'])

rule28 = ctrl.Rule(voltage_ctrl['Low'] & temperature_ctrl['Low'], output_ctrl['Decrease'])
rule29 = ctrl.Rule(voltage_ctrl['Low'] & temperature_ctrl['High'], output_ctrl['Increase'])
rule30 = ctrl.Rule(voltage_ctrl['High'] & temperature_ctrl['Low'], output_ctrl['Increase'])
rule31 = ctrl.Rule(voltage_ctrl['High'] & temperature_ctrl['High'], output_ctrl['Decrease'])

rule32 = ctrl.Rule(temperature_ctrl['Low'] & humidity_ctrl['Low'], output_ctrl['Decrease'])
rule33 = ctrl.Rule(temperature_ctrl['High'] & humidity_ctrl['High'], output_ctrl['Decrease'])
rule34 = ctrl.Rule(temperature_ctrl['Medium'] & humidity_ctrl['Medium'], output_ctrl['No Change'])

rule35 = ctrl.Rule(voltage_ctrl['Medium'] & humidity_ctrl['Low'], output_ctrl['No Change'])
rule36 = ctrl.Rule(voltage_ctrl['Medium'] & humidity_ctrl['High'], output_ctrl['Increase'])
rule37 = ctrl.Rule(voltage_ctrl['Low'] & temperature_ctrl['Medium'], output_ctrl['No Change'])
rule38 = ctrl.Rule(voltage_ctrl['High'] & temperature_ctrl['Medium'], output_ctrl['Increase'])

rule39 = ctrl.Rule(humidity_ctrl['Low'], output_ctrl['Decrease'])
rule40 = ctrl.Rule(humidity_ctrl['High'], output_ctrl['Increase'])

# Collect all 40 rules in the rule base
rule_base = [
    rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9,
    rule10, rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18,
    rule19, rule20, rule21, rule22, rule23, rule24, rule25, rule26, rule27,
    rule28, rule29, rule30, rule31, rule32, rule33, rule34,
    rule35, rule36, rule37, rule38, rule39, rule40
]

# Step 3: Create control system
nfis_ctrl_system = ctrl.ControlSystem(rule_base)
nfis_simulation = ctrl.ControlSystemSimulation(nfis_ctrl_system)

# Step 4: Train using gradient descent and cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
errors = []

for train_index, test_index in kf.split(dataset):
    train_data = dataset.iloc[train_index]
    test_data = dataset.iloc[test_index]

    # Simulate NFIS for training and testing
    predictions = []
    actuals = []

    for i, row in test_data.iterrows():
        nfis_simulation.input['Voltage'] = row['Voltage (V)']
        nfis_simulation.input['Temperature'] = row['Temperature (°C)']
        nfis_simulation.input['Humidity'] = row['Humidity (%RH)']
        nfis_simulation.compute()

        predictions.append(nfis_simulation.output['Output Adjustment'])
        actuals.append(row['Output Adjustment (N)'])

    mse = mean_squared_error(actuals, predictions)
    errors.append(mse)

# Average MSE over folds
average_mse = np.mean(errors)
print(f"Average Mean Squared Error (MSE) from Cross-Validation: {average_mse:.4f}")

# Step 5: Plot membership functions and save them
output_dir = "E:\\Piezoelectric\\plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Voltage membership function plot
voltage_ctrl.view()
plt.savefig(os.path.join(output_dir, 'Voltage_Membership_Functions.png'))

# Temperature membership function plot
temperature_ctrl.view()
plt.savefig(os.path.join(output_dir, 'Temperature_Membership_Functions.png'))

# Humidity membership function plot
humidity_ctrl.view()
plt.savefig(os.path.join(output_dir, 'Humidity_Membership_Functions.png'))

# Output adjustment membership function plot
output_ctrl.view()
plt.savefig(os.path.join(output_dir, 'Output_Adjustment_Membership_Functions.png'))

print(f"Plots saved in directory: {output_dir}")
