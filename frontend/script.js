document.addEventListener('DOMContentLoaded', () => {
    const generateBtn = document.getElementById('generateBtn');
    const loader = document.getElementById('loader');
    const resultsContainer = document.getElementById('results-container');
    const welcomeContainer = document.getElementById('welcome-container'); // Get the new welcome container
    
    const reportOutput = document.getElementById('reportOutput');
    const kpiContainer = document.getElementById('kpi-container');
    const chartsContainer = document.getElementById('charts-container');
    
    const converter = new showdown.Converter({ tables: true, strikethrough: true, simpleLineBreaks: true });
    let chartInstances = {};

    function renderDashboard(dashboardData) {
        kpiContainer.innerHTML = '';
        chartsContainer.innerHTML = '';
        Object.values(chartInstances).forEach(chart => chart.destroy());
        chartInstances = {};

        if (!dashboardData || typeof dashboardData !== 'object' || Object.keys(dashboardData).length === 0) {
            kpiContainer.innerHTML = `<div class="col-12"><div class="alert alert-warning" style="background-color: var(--card-color); border-color: var(--border-color); color: var(--text-color);">Could not generate dashboard visuals. The AI may have failed to produce valid data.</div></div>`;
            return;
        }

        // Render KPI Cards
        if (dashboardData.kpiCards && dashboardData.kpiCards.length > 0) {
            dashboardData.kpiCards.forEach(kpi => {
                const kpiCol = document.createElement('div');
                kpiCol.className = 'col-lg-3 col-md-6';
                kpiCol.innerHTML = `<div class="kpi-card h-100"><div class="kpi-title">${kpi.title}</div><div class="kpi-value">${kpi.value}</div></div>`;
                kpiContainer.appendChild(kpiCol);
            });
        }

        // Render Charts Dynamically
        const chartColors = ['#007bff', '#6f42c1', '#198754', '#ffc107', '#dc3545', '#0dcaf0'];
        Chart.defaults.color = 'rgba(240, 240, 240, 0.8)';
        const chartOptions = {
            scales: {
                y: { beginAtZero: true, ticks: { color: 'rgba(240, 240, 240, 0.8)' }, grid: { color: 'rgba(255, 255, 255, 0.1)' } },
                x: { ticks: { color: 'rgba(240, 240, 240, 0.8)' }, grid: { color: 'rgba(255, 255, 255, 0.1)' } }
            },
            plugins: { legend: { labels: { color: 'rgba(240, 240, 240, 0.8)' } } },
            responsive: true,
            maintainAspectRatio: false
        };

        if (dashboardData.barChart && dashboardData.barChart.labels && dashboardData.barChart.data) {
            const chartCol = document.createElement('div');
            chartCol.className = 'col-lg-8';
            chartCol.innerHTML = `<div class="chart-card"><div class="chart-title">${dashboardData.barChart.title || 'Bar Chart'}</div><canvas id="barChartCanvas"></canvas></div>`;
            chartsContainer.appendChild(chartCol);
            const barCtx = document.getElementById('barChartCanvas').getContext('2d');
            chartInstances.bar = new Chart(barCtx, { type: 'bar', data: { labels: dashboardData.barChart.labels, datasets: [{ label: 'Total', data: dashboardData.barChart.data, backgroundColor: chartColors[0] }] }, options: chartOptions });
        }
        if (dashboardData.pieChart && dashboardData.pieChart.labels && dashboardData.pieChart.data) {
            const chartCol = document.createElement('div');
            chartCol.className = 'col-lg-4';
            chartCol.innerHTML = `<div class="chart-card"><div class="chart-title">${dashboardData.pieChart.title || 'Pie Chart'}</div><canvas id="pieChartCanvas"></canvas></div>`;
            chartsContainer.appendChild(chartCol);
            const pieCtx = document.getElementById('pieChartCanvas').getContext('2d');
            chartInstances.pie = new Chart(pieCtx, { type: 'pie', data: { labels: dashboardData.pieChart.labels, datasets: [{ data: dashboardData.pieChart.data, backgroundColor: chartColors }] }, options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: 'rgba(240, 240, 240, 0.8)' } } } } });
        }
    }

    generateBtn.addEventListener('click', async () => {
        const file = document.getElementById('fileUpload').files[0];
        if (!file) { alert('Please select a file first.'); return; }

        // --- Robust State Management ---
        generateBtn.disabled = true;
        generateBtn.innerHTML = `<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Analyzing...`;
        welcomeContainer.classList.add('d-none'); // Hide welcome message
        loader.classList.remove('d-none');
        resultsContainer.classList.add('d-none');
        
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/generate-report/', { method: 'POST', body: formData });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(`HTTP error! status: ${response.status}, detail: ${errorData.detail}`);
            }

            const data = await response.json();
            
            if (typeof data.report === 'string' && data.report.trim() !== '') {
                reportOutput.innerHTML = converter.makeHtml(data.report);
            } else {
                reportOutput.innerHTML = `<div class="alert alert-warning">The AI failed to generate a text report.</div>`;
            }
            
            renderDashboard(data.dashboardData);
            
            resultsContainer.classList.remove('d-none');

        } catch (error) {
            console.error('Error fetching dashboard:', error);
            const errorHtml = converter.makeHtml(`## Analysis Failed\n\nA critical error occurred:\n\n\`\`\`\n${error.message}\n\`\`\``);
            reportOutput.innerHTML = `<div class="alert alert-danger" style="background-color: var(--card-color); border-color: var(--border-color); color: var(--text-color);">${errorHtml}</div>`;
            new bootstrap.Tab(document.getElementById('analysis-tab')).show();
            resultsContainer.classList.remove('d-none');
        } finally {
            // This block is guaranteed to run, permanently fixing the "stuck button"
            generateBtn.disabled = false;
            generateBtn.innerHTML = 'Generate Dashboard';
            loader.classList.add('d-none');
        }
    });
});

