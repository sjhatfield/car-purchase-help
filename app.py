from functools import lru_cache

from flask import Flask, render_template, request

import car_purchase_help.model1 as model_1
from car_purchase_help.utils import format_user_input, clean_input

app = Flask(__name__)


@app.route("/")
def landing_page():
    """
    Renders landing page
    """
    return render_template("landing.html")


@app.route("/model1", methods=["POST", "GET"])
def model1():
    """
    Renders model1 input form and results
    """
    return handle_request(request, "model1.html")


def get_model_from_template(template_name):
    """
    Get the name of the relevant model from the name of the template
    :param template_name: name of html template
    :return: name of the model
    """
    return template_name.split(".")[0]


@lru_cache(maxsize=128)
def retrieve_advice_from_model(user_raw_text, model):
    """
    This function computes or retrieves advice
    We use an LRU cache to store results we process. If we see the same car details
    twice, we can retrieve cached results to serve them faster
    :param user_raw_text: the input text to the model eg honda, camry, 2013, 30000, 4000
    :param model: which model to use
    :return: a model's recommendations
    """
    if model == "model1":
        # Get the car details from the raw text that the user entered
        manufacturer, model, year, odometer, listed_price = format_user_input(
            user_raw_text
        )
        print(
            f"formatted user input: {manufacturer}, {model}, {year}, {odometer}, {listed_price}"
        )
        predicted_price, mean_absolute_residual = model_1.predict_price(
            manufacturer, model, year, odometer
        )
        print(
            f"Predicted price and residual {predicted_price}, {mean_absolute_residual}"
        )
        # Return the advice given the prediciton and listed price
        return model_1.get_advice(predicted_price, listed_price, mean_absolute_residual)

    raise ValueError("Incorrect Model passed")


def handle_request(request, template_name):
    """
    Renders an input form for GET requests and displays results for the given
    posted question for a POST request
    :param request: http request
    :param template_name: name of the requested template (e.g. model1.html)
    :return: Render an input form or results depending on request type
    """
    if request.method == "POST":
        # Rertrieve the text that the user entered
        car_info = request.form.get("car_info")
        # Retrieve the model the user selected to use
        model_name = get_model_from_template(template_name)
        # Attempt to get advice according to the model
        try:
            advice = retrieve_advice_from_model(car_info, model_name)
            print(advice)
            payload = {
                "input": car_info,
                "advice": advice,
                "model_name": model_name,
                "error": "",
            }
            return render_template("results.html", ml_result=payload)

        # There are various reasons this can fail. If it does render the error
        # page and let the user know what went wrong
        except Exception as error:
            print(error)
            payload = {
                "input": car_info,
                "advice": "",
                "model_name": model_name,
                "error": error,
            }
            return render_template("error.html", ml_result=payload)

    else:
        return render_template(template_name)
