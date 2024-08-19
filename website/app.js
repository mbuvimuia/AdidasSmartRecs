document.getElementById('recommendation-form').addEventListener('submit', async function (e) {
    e.preventDefault();

    const userId = document.getElementById('user-id').value;
    const nRecommendations = 5;  // or any number you want

    try {
        const response = await fetch(`http://127.0.0.1:8000/recommendations/?user_id=${userId}&n_recommendations=${nRecommendations}`);
        const data = await response.json();

        const recommendationList = document.getElementById('recommendation-list');
        recommendationList.innerHTML = '';

        data.recommendations.forEach(item => {
            const listItem = document.createElement('li');
            listItem.textContent = `Product ID: ${item}`;
            recommendationList.appendChild(listItem);
        });
    } catch (error) {
        console.error('Error fetching recommendations:', error);
    }
});
