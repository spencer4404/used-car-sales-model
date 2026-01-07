const form = document.getElementById("predict-form");
const result = document.getElementById("result");

// define the arrays users can select from
const CONDS = ['Good', 'Excellent', 'New', 'Fair', 'Like new', 'Salvage']; // conditions
const DRIVES = ['Rwd', '4wd', 'Fwd']; // rear, 4, front wheel drive
const STATES = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 
    'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 
    'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NC', 'NE', 'NV', 
    'NJ', 'NM', 'NY', 'NH', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 
    'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
    ];
const COLORS = ['Black', 'Silver', 'Grey', 'Red', 'Blue', 'White', 'Brown', 'Yellow', 'Green', 'Orange', 'Purple', 'Custom',]
const TYPE = ['Sedan', 'Coupe', 'Suv', 'Truck', 'Pickup', 'Other', 'Hatchback', 'Mini-van', 'Offroad', 'Convertible', 'Wagon', 'Van', 'Bus']
const FUEL = ['Gas', 'Hybrid', 'Electric', 'Diesel', 'Other']

// populate each dropdown
const DROPDOWNS = {
    condition: CONDS,
    drive: DRIVES,
    state: STATES,
    color: COLORS,
    type: TYPE,
    fuel: FUEL
};

Object.entries(DROPDOWNS).forEach(([id,values]) => {
    // select each id
    const select = document.getElementById(id);
    // get each of the values
    values.forEach(v =>{
        const opt = document.createElement("option");
        opt.value = v;
        opt.textContent = v;
        select.appendChild(opt);
    })
})


form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const payload = {
    age: Number(document.getElementById("age").value),
    manufacturer: document.getElementById("manufacturer").value.toLowerCase(),
    model: document.getElementById("model").value.toLowerCase(),
    trim: document.getElementById("trim").value.toLowerCase(),
    condition: document.getElementById("condition").value.toLowerCase(),
    fuel: document.getElementById("fuel").value.toLowerCase(),
    odometer: Number(document.getElementById("odometer").value),
    drive: document.getElementById("drive").value.toLowerCase(),
    type: document.getElementById("type").value.toLowerCase(),
    paint_color: document.getElementById("color").value.toLowerCase(),
    state: document.getElementById("state").value.toLowerCase(),
    lat: 42.0,
    long: -71.0
  };

  const res = await fetch("http://localhost:8000/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  const data = await res.json();
  result.textContent = `Estimated Price: $${data.predicted_price.toFixed(0)}`;
});