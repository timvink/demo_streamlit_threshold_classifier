# Using streamlit to set the threshold for a classifier

Machine learning classifiers can make binary predictions by setting a threshold on the predicted probabilities. This threshold defaults at `>=0.5`, but many business problems benefit from thoughtful adaptation of this threshold. This is especially true for unbalanced machine learning problems. Changing the threshold is inherently a tradeoff between [precision](https://en.wikipedia.org/wiki/Precision_and_recall) and [recall](https://en.wikipedia.org/wiki/Precision_and_recall) and should be done together with business stakeholders that understand the problem domain.

[streamlit](https://docs.streamlit.io/en/stable/index.html) is an open source python library that makes it easy to build a custom webapp.

This repository demonstrates a streamlit app that can facilitate the interactive setting of thresholds together with the business. Here's a quick preview:

## Example model

We generate a dataset `X` with 30k observations, 20 features and a class imbalance of 9:1.
We use a stratified `train_test_split` with 80% train and 20% test.

We then train a simple `RandomForestClassifier` model on the train set, using a 5-fold cross-validated grid search to tune the hyperparameters.

## App

To set the threshold for our classifier dynamically, run the app with:

```bash
pip install -r requirements.txt
streamlit run app.py
```

This is what it looks like:

<img src="demo.gif" style="max-height: 400px" />

## Further reading

- [Streamlit video tutorial](https://calmcode.io/streamlit/hello-world.html) Crystal-clear and concise video tutorial by [calmcode.io](https://calmcode.io/)
- [Streamlit 101: An in-depth introduction](https://towardsdatascience.com/streamlit-101-an-in-depth-introduction-fc8aad9492f2) Great example use-case on NYC airbnb data
- [Streamlit API reference](https://docs.streamlit.io/en/stable/api.html#display-text) Overview of all the streamlist elements
- [Streamlit community components](https://www.streamlit.io/components) 
- [awesome-streamlit](https://github.com/MarcSkovMadsen/awesome-streamlit) list of streamlit resources
- [github streamlit topic](https://github.com/topics/streamlit) is a great way to discover more streamlit libraries