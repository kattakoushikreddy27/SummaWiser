// DOM Elements
const inputText = document.getElementById('input-text');
const processBtn = document.getElementById('process-btn');
const clearBtn = document.getElementById('clear-btn');
const summaryOutput = document.getElementById('summary-output');
const questionInput = document.getElementById('question-input');
const askBtn = document.getElementById('ask-btn');
const qaOutput = document.getElementById('qa-output');
const loading = document.getElementById('loading');
const tabButtons = document.querySelectorAll('.tab-button');
const resultTabs = document.querySelectorAll('.result-tab');

// Tab switching functionality
tabButtons.forEach(button => {
    button.addEventListener('click', () => {
        const tabName = button.getAttribute('data-tab');

        // Update active tab button
        tabButtons.forEach(btn => btn.classList.remove('active'));
        button.classList.add('active');

        // Show corresponding tab content
        resultTabs.forEach(tab => tab.classList.remove('active'));
        document.getElementById(`${tabName}-tab`).classList.add('active');
    });
});

// Process text button click handler
processBtn.addEventListener('click', () => {
    const text = inputText.value.trim();

    if (text === '') {
        alert('Please enter some text to process.');
        return;
    }

    // Show loading indicator
    loading.style.display = 'block';
    summaryOutput.style.display = 'none';

    // Call the summarize API
    fetch('/api/summarize', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: text }),
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        let summaryHTML = '<div class="summary-list">';

        data.summary_points.forEach((point, index) => {
            summaryHTML += `
                <div class="summary-point">
                    <span class="summary-point-number">${index + 1}</span>
                    ${point}
                </div>
            `;
        });

        summaryHTML += '</div>';
        summaryOutput.innerHTML = summaryHTML;

        // Hide loading indicator
        loading.style.display = 'none';
        summaryOutput.style.display = 'block';

        // Switch to summary tab
        tabButtons.forEach(btn => btn.classList.remove('active'));
        tabButtons[0].classList.add('active');
        resultTabs.forEach(tab => tab.classList.remove('active'));
        document.getElementById('summary-tab').classList.add('active');
    })
    .catch(error => {
        console.error('Error:', error);
        summaryOutput.innerHTML = '<p>Error generating summary. Please try again.</p>';
        loading.style.display = 'none';
        summaryOutput.style.display = 'block';
    });
});

// Ask question button click handler
askBtn.addEventListener('click', () => {
    const question = questionInput.value.trim();
    const text = inputText.value.trim();

    if (question === '') {
        alert('Please enter a question.');
        return;
    }

    if (text === '') {
        alert('Please process some text first.');
        return;
    }

    // Show loading indicator
    loading.style.display = 'block';
    qaOutput.style.display = 'none';

    // Call the question answering API
    fetch('/api/answer', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            question: question,
            text: text
        }),
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        qaOutput.innerHTML = `
            <p><strong>Q: ${question}</strong></p>
            <p>${data.answer}</p>
        `;

        // Hide loading indicator
        loading.style.display = 'none';
        qaOutput.style.display = 'block';
    })
    .catch(error => {
        console.error('Error:', error);
        qaOutput.innerHTML = '<p>Error generating answer. Please try again.</p>';
        loading.style.display = 'none';
        qaOutput.style.display = 'block';
    });
});

// Clear button click handler
clearBtn.addEventListener('click', () => {
    inputText.value = '';
    summaryOutput.innerHTML = '<p>Your summary will appear here after processing.</p>';
    qaOutput.innerHTML = '<p>Ask a question about your text to get an answer.</p>';
    questionInput.value = '';
});