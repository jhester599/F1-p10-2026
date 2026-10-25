from src.openf1 import qualifying_rows_from_openf1, starting_grid_from_openf1


def test_unknown_car_number_gets_teammate_constructor_and_surname_id() -> None:
    results = [
        {"position": 1, "driver_number": 30, "duration": [95.0, 94.5, 94.1]},
        {"position": 2, "driver_number": 99, "duration": [95.2, 94.9, None]},
        {"position": None, "driver_number": 31, "duration": [None, None, None]},
    ]
    drivers = [
        {"driver_number": 30, "team_name": "Racing Bulls", "last_name": "Lawson"},
        {"driver_number": 99, "team_name": "Racing Bulls", "last_name": "Iwasa"},
        {"driver_number": 31, "team_name": "Haas F1 Team", "last_name": "Ocon"},
    ]
    number_map = {"30": ("lawson", "rb"), "31": ("ocon", "haas")}

    rows = qualifying_rows_from_openf1(results, drivers, number_map)

    assert [r["Driver"]["driverId"] for r in rows] == ["lawson", "iwasa", "ocon"]
    assert rows[1]["Constructor"]["constructorId"] == "rb"
    assert rows[1]["Q2"] == "1:34.900"
    assert "Q3" not in rows[1]
    # Unclassified entries are placed at the back without lap times.
    assert rows[2]["position"] == "3"
    assert "Q1" not in rows[2]


def test_starting_grid_maps_car_numbers_to_driver_ids() -> None:
    grid = [{"driver_number": 30, "position": 14}, {"driver_number": 7, "position": 1}]

    assert starting_grid_from_openf1(grid, {"30": ("lawson", "rb")}) == {"lawson": 14}
