# stroke_prediction

- An interactive web application to predict stroke by machine learning built with https://streamlit.io/.
- Data visualization and analysis with multiple machine learning models.

## G66_source_code.ipynb
This notebook contains the visualization of the [stroke data](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset) together with four machine learning models - **Logistic Regression, Gaussian Naive Bayes, k-nearest neighbors, and Artificial neural network with Genetic Algorithm optimization**- training on it. The dataset being used in this program is `healthcare-dataset-stroke-data.csv`. This is the original dataset directly downloaded from the aforementioned kaggle website.

## app.py
This is an interative web app to predict stroke. The dataset being used in this program is `stroke.csv`. This is the processed data containing only numerical values.

### Usage & 
- Install *streamlit* on your local machine by `pip install streamlit`.
- To run the app, simply run the command: `streamlit run app.py`.

### Description
- You may tweak the values at the sidebars to try out different factors that might cause stroke.
- The prediction will be shown in the **Classification** section.

![alt text](https://github.com/chiatsekuo/stroke_prediction/blob/main/screenshots/landing_page.PNG "app landing page")
![alt text](https://github.com/chiatsekuo/stroke_prediction/blob/main/screenshots/second_scroll.PNG "app landing page")
