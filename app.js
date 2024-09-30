// app.js

// Wait for the DOM to load
document.addEventListener("DOMContentLoaded", function() {
    const form = document.getElementById("loan-form");
    const resultDiv = document.getElementById("result");
    const explanationDiv = document.getElementById("explanation");

    // Event listener for form submission
    form.addEventListener("submit", function(event) {
        event.preventDefault(); // Prevent default form submission

        // Collect input data
        const formData = new FormData(form);
        const data = {
            loan_amount: formData.get("loan_amount"),
            income: formData.get("income"),
            credit_score: formData.get("credit_score"),
            employment_status: formData.get("employment_status")
        };

        // Send data to the Flask backend via AJAX
        fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(data => {
            // Display the prediction result
            resultDiv.innerHTML = `<h3>Loan Approval: ${data.approval}</h3>`;

            // Display the explanation
            explanationDiv.innerHTML = `<h4>Explanation:</h4><pre>${JSON.stringify(data.explanation, null, 2)}</pre>`;
        })
        .catch(error => {
            console.error("Error:", error);
            resultDiv.innerHTML = `<h3>An error occurred. Please try again.</h3>`;
        });
    });
});
