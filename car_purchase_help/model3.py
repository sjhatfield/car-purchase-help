import ast
from pathlib import Path

from car_purchase_help.model1 import predict_price
from car_purchase_help.model2 import predict_mileage_cost


def get_similar_advice(manufacturer: str, model: str, year: int, odometer: int) -> str:
    """
    For a vechicle looks up vehicles classed as similar and returns their predicted price
    and information about mileage reduction from model1 and model2
    :param df: dataframe containing the data (assuemd to be clean and formatted)
    :param manufacturer: manufacturer of the vechicle
    :param model: model of the vehicle
    :param year: year of manufacture
    :return: string of advice
    """
    similar_cars_path = Path(f"data/clean_similar_cars.txt")
    with open(similar_cars_path, "r") as f:
        similar_cars_dict = ast.literal_eval(f.read())
    ret_str = (
        "<br><br><br><b>Here is the same information but for similar vehicles "
        "for the same year and mileage</b><br>"
    )

    try:
        similar_cars = similar_cars_dict[f"{manufacturer}_{model}"]
    except:
        return ret_str + "<br/>Sorry no similar vehicles were available"

    for car in similar_cars:
        car_manu = car.split("_")[0]
        car_model = car.split("_")[1]
        try:
            price, _ = predict_price(car_manu, car_model, year, odometer)
            mileage_advice = predict_mileage_cost(car_manu, car_model, year)
            ret_str += f"<b>{car_manu.title()}</b>, <b>{car_model.title()}</b>: would cost approximately ${round(price, 2)}. {mileage_advice}<br/>"
        except:
            pass
    return ret_str
