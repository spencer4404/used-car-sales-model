const form = document.getElementById("predict-form");
const result = document.getElementById("result");

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const payload = {
    age: Number(document.getElementById("age").value),
    manufacturer: document.getElementById("manufacturer").value,
    model: document.getElementById("model").value,
    trim: "",
    condition: "good",
    fuel: "gas",
    odometer: Number(document.getElementById("odometer").value),
    drive: "4wd",
    type: "suv",
    paint_color: "black",
    state: "ma",
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