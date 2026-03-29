const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const loading = document.getElementById('loading');
const resultsContainer = document.getElementById('results-container');
const resultsBody = document.getElementById('results-body');
const resetBtn = document.getElementById('reset-btn');

// Drag and drop events
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
});

dropZone.addEventListener('drop', handleDrop, false);
fileInput.addEventListener('change', handleFiles, false);

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles({ target: { files: files } });
}

function handleFiles(e) {
    const file = e.target.files[0];
    if (file && file.name.endsWith('.csv')) {
        uploadFile(file);
    } else {
        alert("Please upload a valid CSV file.");
    }
}

async function uploadFile(file) {
    dropZone.classList.add('hidden');
    loading.classList.remove('hidden');

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/predict_batch', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            renderResults(data.predictions);
        } else {
            alert("Error: " + data.error);
            resetView();
        }
    } catch (error) {
        alert("Network error occurred: " + error);
        resetView();
    }
}

function renderResults(predictions) {
    loading.classList.add('hidden');
    resultsContainer.classList.remove('hidden');
    
    resultsBody.innerHTML = '';
    
    predictions.forEach(pred => {
        const tr = document.createElement('tr');
        
        // Format price
        const formattedPrice = new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD'
        }).format(pred.price);

        const isAbove = pred.category.includes('Above');

        tr.innerHTML = `
            <td>#${pred.id}</td>
            <td style="font-weight: 600; color: #60a5fa">${formattedPrice}</td>
            <td>
                <span style="padding: 4px 8px; border-radius: 4px; font-size: 0.85rem; background: ${isAbove ? 'rgba(16, 185, 129, 0.2)' : 'rgba(245, 158, 11, 0.2)'}; color: ${isAbove ? '#34d399' : '#fbbf24'}">
                    ${pred.category}
                </span>
            </td>
        `;
        resultsBody.appendChild(tr);
    });
}

function resetView() {
    resultsContainer.classList.add('hidden');
    loading.classList.add('hidden');
    dropZone.classList.remove('hidden');
    fileInput.value = ''; // clear input
}

resetBtn.addEventListener('click', resetView);
