document.getElementById('get-recommendations').addEventListener('click', async function() {
    const userId = document.getElementById('user-id').value;
    const response = await fetch(`/recommendations/${userId}`);
    
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

    recommendations.forEach(product => {
        const productDiv = document.createElement('div');
        productDiv.className = 'product-item';
        productDiv.textContent = `Product ID: ${product}`;
        productList.appendChild(productDiv);
    });
}