CREATE TABLE user_input_data (
    id SERIAL PRIMARY KEY,

    age INT,
    manufacturer TEXT,
    model TEXT,
    trim TEXT,
    condition TEXT,
    fuel TEXT,
    odometer INT,
    drive TEXT,
    vehicle_type TEXT,
    color TEXT,
    state TEXT,
    lat DOUBLE PRECISION,
    long DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE predictions(
    id SERIAL PRIMARY KEY,
    input_id INT REFERENCES user_input_data(id),
    predicted_price DOUBLE PRECISION,
    model_version TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);