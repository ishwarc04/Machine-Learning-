document.getElementById('prediction-form').addEventListener('submit', function(event) {
    event.preventDefault();
    const level = document.getElementById('level').value;
    const resultDiv = document.getElementById('result');

    // Make a request to the Django backend
    fetch('/predict/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            // Get the CSRF token from a meta tag or a hidden input field
            'X-CSRFToken': '{{ csrf_token }}' // This requires Django template syntax
        },
        body: JSON.stringify({ level: parseFloat(level) })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            resultDiv.textContent = `Error: ${data.error}`;
            resultDiv.style.color = 'red';
        } else {
            resultDiv.textContent = `Predicted Salary: $${data.salary.toFixed(2)}`;
            resultDiv.style.color = '#28a745';
        }
    })
    .catch(error => {
        console.error('Error:', error);
        resultDiv.textContent = 'An error occurred. Please try again.';
        resultDiv.style.color = 'red';
    });
});