import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap

# Load the trained model
model = joblib.load('hemaa.pkl')

# Streamlit app title
st.title('Net Profit Prediction')

# User input section for features
st.sidebar.header('Input Features')
start_year = st.sidebar.number_input('Start Year', min_value=2000, max_value=2100, step=1)
end_year = st.sidebar.number_input('End Year', min_value=2000, max_value=2100, step=1)
future_years = st.sidebar.number_input('Years into the Future', min_value=0, max_value=100, step=1)
production_volume = st.sidebar.number_input('Total net production volume (kg)')
expected_price = st.sidebar.number_input('Expected price (Euro/Kg)')
revenue = st.sidebar.number_input('Revenue (Euro)')
fixed_cost = st.sidebar.number_input('Yearly Fixed cost')
variable_cost = st.sidebar.number_input('Variable cost')
cash_flow = st.sidebar.number_input('Cash Flow')

# Generate predictions for each year in the range
years = list(range(start_year, end_year + 1)) + list(range(end_year + 1, end_year + 1 + future_years))
predictions = []

for year in years:
    input_data = pd.DataFrame({
        'Year': [year],
        'Total net production volume (kg)': [production_volume],
        'Expected price (Euro/Kg)': [expected_price],
        'Revenue (Euro)': [revenue],
        'Yearly Fixed cost': [fixed_cost],
        'Variable cost': [variable_cost],
        'Cash Flow': [cash_flow]
    })
    prediction = model.predict(input_data)
    predictions.append(prediction[0])

# Display predictions
if st.sidebar.button('Predict'):
    st.write(f'Predicted Net Profit from {start_year} to {end_year + future_years}')
    for year, pred in zip(years, predictions):
        st.write(f'Year {year}: €{pred:,.2f}')

    # Plotting the predictions
    fig, ax = plt.subplots()
    ax.plot(years, predictions, label='Predicted Net Profit', marker='o')
    ax.set_xlabel('Year')
    ax.set_ylabel('Net Profit (€)')
    ax.set_title('Net Profit Prediction Over Time')
    ax.legend()
    st.pyplot(fig)

    # Feature importance analysis using SHAP
    st.write("### Feature Importance Analysis")
    input_data = pd.DataFrame({
        'Year': [start_year],
        'Total net production volume (kg)': [production_volume],
        'Expected price (Euro/Kg)': [expected_price],
        'Revenue (Euro)': [revenue],
        'Yearly Fixed cost': [fixed_cost],
        'Variable cost': [variable_cost],
        'Cash Flow': [cash_flow]
    })
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)
    shap.summary_plot(shap_values, input_data, plot_type="bar")
    st.pyplot(bbox_inches='tight')
