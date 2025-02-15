import argparse
from random import choice, randint
from tenacity import retry, stop_after_attempt, wait_exponential
import requests as Client

# Definir argumentos con argparse
parser = argparse.ArgumentParser(
    description="Test API de predicción de precios de autos."
)

parser.add_argument(
    "--api_url",
    required=True,
    type=str,
    help="URL de la API a probar",
)

args = parser.parse_args()

# **Datos de prueba**
models = ["BMW", "Tesla", "Cadillac", "Land", "GMC", "Toyota", "Hyundai"]
colors = ["Blue", "White", "Red", "Brown", "Dark Galvanized"]
fuel_types = ["Gasoline", "E85 Flex Fuel", "nan", "Hybrid", "Diesel"]
transmissions = [
    "5-Speed Automatic",
    "2-Speed Automatic",
    "8-SPEED A/T",
    "7-Speed",
    "Variable",
]
engines = [
    "425.0HP 3.0L Straight 6 Cylinder Engine Gasoline Fuel",
    "312.0HP 3.6L V6 Cylinder Engine Gasoline Fuel",
    "Electric Motor Electric Fuel System",
    "420.0HP 6.2L 8 Cylinder Engine Gasoline Fuel",
]
accidents = ["None reported", "At least 1 accident or damage reported"]


def generate_test_data() -> dict:
    return {
        "brand":choice(models),
        "model":choice(models),
        "accident":choice(accidents),
        "clean_title":choice([True, False]),
        "engine":choice(engines),
        "ext_col":choice(colors),
        "int_col":choice(colors),
        "fuel_type":choice(fuel_types),
        "milage":randint(0, 100000),
        "model_year":randint(1974, 2023),
        "transmission":choice(transmissions),
    }


def print_request_result(res, title):
    if res.status_code == 200:
        print("✅ ", title, " - OK")

    else:
        print("❌ ", title, " - FAIL, status code: ", res.status_code)
    print(res.json(), end="\n\n")
    

@retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=3, min=4, max=20),
        reraise=True,
    )
def test(api_url: str) -> None:
    print("Testing API - Predict")
    test_data = generate_test_data()
    res = Client.post(f"{api_url}/predict", json=test_data)
    print_request_result(res, "Predict")

    print("Test API - Train")
    res = Client.post(f"{api_url}/train")
    print_request_result(res, "Train")


if __name__ == "__main__":
    test(args.api_url)