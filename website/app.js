document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('recommendation-form');
    const recommendationsDiv = document.getElementById('recommendations');

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        const userId = document.getElementById('user-id').value.trim();
        recommendationsDiv.innerHTML = '<em>Loading recommendations...</em>';
        try {
            const response = await fetch(`/recommendations/${userId}`);
            if (response.ok) {
                const data = await response.json();
                displayRecommendations(data.recommendations);
            } else {
                recommendationsDiv.innerHTML = '<span style="color:#f55">No recommendations found for this user.</span>';
            }
        } catch (err) {
            recommendationsDiv.innerHTML = '<span style="color:#f55">Error fetching recommendations.</span>';
        }
    });

    function displayRecommendations(recommendations) {
        recommendationsDiv.innerHTML = '';
        if (!recommendations || recommendations.length === 0) {
            recommendationsDiv.textContent = 'No recommendations found.';
            return;
        }
        const ul = document.createElement('ul');
        recommendations.forEach(productId => {
            const li = document.createElement('li');
            li.textContent = `Product ID: ${productId}`;
            ul.appendChild(li);
        });
        recommendationsDiv.appendChild(ul);
    }
});

