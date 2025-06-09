document.getElementById('get-recommendations').addEventListener('click', async function() {
    const userId = document.getElementById('user-id').value;
    const response = await fetch(`http://127.0.0.1:8000/recommendations/${userId}`);
    if (response.ok) {
        const data = await response.json();
        displayRecommendations(data.recommendations);
    } else {
        alert('No recommendations found for this user.');
    }
});

function displayRecommendations(recommendations) {
    const productList = document.getElementById('product-list');
    productList.innerHTML = ''; // Clear any previous recommendations

    if (recommendations.length === 0) {
        productList.textContent = 'No recommendations found.';
        return;
    }

    recommendations.forEach(product => {
        const productDiv = document.createElement('div');
        productDiv.className = 'product-item';
        productDiv.textContent = `Product: ${product[0]}, Category: ${product[1]}`;
        productList.appendChild(productDiv);
    });
}

