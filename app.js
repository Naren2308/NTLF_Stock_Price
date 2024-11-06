async function getPrediction() {
    const pricesText = document.getElementById("prices").value;
    const pricesArray = pricesText.split(",").map(Number);

    if (pricesArray.length !== 60) {
        document.getElementById("result").innerText = "Please enter exactly 60 prices.";
        return;
    }

    // Fetch prediction from Flask API
    const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prices: pricesArray })
    });

    const data = await response.json();
    if (response.ok) {
        document.getElementById("result").innerText = `Predicted Price: $${data.predicted_price.toFixed(2)}`;
        displayChart(pricesArray, data.predicted_price);
    } else {
        document.getElementById("result").innerText = "Error: " + data.error;
    }
}

function displayChart(pricesArray, predictedPrice) {
    const ctx = document.getElementById('priceChart').getContext('2d');
    const labels = Array.from({ length: pricesArray.length + 1 }, (_, i) => i + 1);
    const data = [...pricesArray, predictedPrice];

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Price Prediction',
                data: data,
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 2,
                fill: false
            }]
        },
        options: {
            scales: {
                x: { title: { display: true, text: 'Days' } },
                y: { title: { display: true, text: 'Price (USD)' } }
            }
        }
    });
}
