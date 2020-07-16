import streamlit as st

from model import get_predictions
from eval import get_metrics_df
from vizualization import plot_probability_distribution

# data
@st.cache()
def cached_get_predictions():
    return get_predictions()


y_train, yhat_prob_train, y_test, yhat_prob_test = cached_get_predictions()

# UI
st.title("Setting the threshold for our classifier")
st.write("Interactively using streamlit! :point_down: ")
threshold = st.slider("Threshold", min_value=0.00, max_value=1.0, step=0.01, value=0.5)

# Metrics
metrics = get_metrics_df(
    y_train, yhat_prob_train, y_test, yhat_prob_test, threshold=threshold
)
st.dataframe(metrics.assign(hack="").set_index("hack"))

# Plots
ax = plot_probability_distribution(
    yhat_prob_train, y_train, threshold, "Train predictions"
)
st.pyplot(optimize=True)

ax = plot_probability_distribution(
    yhat_prob_test, y_test, threshold, "Test predictions"
)
st.pyplot(optimize=True)
