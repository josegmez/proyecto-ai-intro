from typing import Literal, Optional

from pydantic import BaseModel, Field


class PredictionData(BaseModel):
    brand: Literal[
        "MINI",
        "Lincoln",
        "Chevrolet",
        "Genesis",
        "Mercedes-Benz",
        "Audi",
        "Ford",
        "BMW",
        "Tesla",
        "Cadillac",
        "Land",
        "GMC",
        "Toyota",
        "Hyundai",
        "Volvo",
        "Volkswagen",
        "Buick",
        "Rivian",
        "RAM",
        "Hummer",
        "Alfa",
        "INFINITI",
        "Jeep",
        "Porsche",
        "McLaren",
        "Honda",
        "Lexus",
        "Dodge",
        "Nissan",
        "Jaguar",
        "Acura",
        "Kia",
        "Mitsubishi",
        "Rolls-Royce",
        "Maserati",
        "Pontiac",
        "Saturn",
        "Bentley",
        "Mazda",
        "Subaru",
        "Ferrari",
        "Aston",
        "Lamborghini",
        "Chrysler",
        "Lucid",
        "Lotus",
        "Scion",
        "smart",
        "Karma",
        "Plymouth",
        "Suzuki",
        "FIAT",
        "Saab",
        "Bugatti",
        "Mercury",
        "Polestar",
        "Maybach",
    ]
    model: str
    model_year: int = Field(..., ge=1974, le=2023)
    milage: int
    fuel_type: Literal[
        "Gasoline",
        "E85 Flex Fuel",
        "nan",
        "Hybrid",
        "Diesel",
        "Plug-In Hybrid",
        "–",
        "not supported",
    ]
    engine: str
    transmission: str
    ext_col: str
    int_col: str
    accident: Optional[
        Literal["None reported", "At least 1 accident or damage reported"]
    ]
    clean_title: bool


class ModelPrediction(BaseModel):
    price: int
