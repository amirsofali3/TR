// Dashboard JavaScript

class TradingDashboard {
    constructor() {
        this.socket = null;
        this.lastUpdate = null;
        this.isConnected = false;
        
        this.init();
    }
    
    init() {
        // Initialize WebSocket connection
        this.initWebSocket();
        
        // Setup event listeners
        this.setupEventListeners();
        
        // Load initial data
        this.loadInitialData();
        
        // Setup periodic updates
        this.setupPeriodicUpdates();
    }
    
    initWebSocket() {
        try {
            this.socket = io();
            
            this.socket.on('connect', () => {
                console.log('Connected to WebSocket');
                this.isConnected = true;
                this.updateConnectionStatus(true);
            });
            
            this.socket.on('disconnect', () => {
                console.log('Disconnected from WebSocket');
                this.isConnected = false;
                this.updateConnectionStatus(false);
            });
            
            this.socket.on('system_update', (data) => {
                this.handleSystemUpdate(data);
            });
            
            this.socket.on('error', (error) => {
                console.error('WebSocket error:', error);
                this.showAlert('WebSocket error: ' + error.message, 'danger');
            });
            
        } catch (error) {
            console.error('Failed to initialize WebSocket:', error);
            this.fallbackToPolling();
        }
    }
    
    fallbackToPolling() {
        // Fallback to HTTP polling if WebSocket fails
        console.log('Falling back to HTTP polling');
        setInterval(() => {
            this.loadSystemStatus();
        }, 5000);
    }
    
    setupEventListeners() {
        // Refresh button
        document.getElementById('refresh-btn').addEventListener('click', () => {
            this.loadInitialData();
        });
        
        // Retrain model button
        document.getElementById('retrain-btn').addEventListener('click', () => {
            this.retrainModel();
        });
        
        // Save TP/SL configuration
        document.getElementById('save-tpsl').addEventListener('click', () => {
            this.saveTpSlConfig();
        });
        
        // Symbol selection change
        document.getElementById('symbol-select').addEventListener('change', (e) => {
            if (e.target.value) {
                this.loadTpSlConfig(e.target.value);
            }
        });
    }
    
    setupPeriodicUpdates() {
        // Update prices every 30 seconds
        setInterval(() => {
            this.loadPrices();
        }, 30000);
        
        // Update portfolio every 60 seconds
        setInterval(() => {
            this.loadPortfolio();
        }, 60000);
    }
    
    async loadInitialData() {
        try {
            this.showLoading(true);
            
            await Promise.all([
                this.loadSystemStatus(),
                this.loadIndicators(),
                this.loadPortfolio(),
                this.loadSignals(),
                this.loadPrices(),
                this.loadSymbolOptions()
            ]);
            
        } catch (error) {
            console.error('Failed to load initial data:', error);
            this.showAlert('Failed to load dashboard data', 'danger');
        } finally {
            this.showLoading(false);
        }
    }
    
    async loadSystemStatus() {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            
            if (response.ok) {
                this.updateSystemStatus(data);
            } else {
                throw new Error(data.error || 'Failed to load system status');
            }
            
        } catch (error) {
            console.error('Failed to load system status:', error);
        }
    }
    
    async loadIndicators() {
        try {
            const response = await fetch('/api/indicators');
            const data = await response.json();
            
            if (response.ok) {
                this.updateIndicators(data);
            } else {
                throw new Error(data.error || 'Failed to load indicators');
            }
            
        } catch (error) {
            console.error('Failed to load indicators:', error);
        }
    }
    
    async loadPortfolio() {
        try {
            const response = await fetch('/api/portfolio');
            const data = await response.json();
            
            if (response.ok) {
                this.updatePortfolio(data);
            } else {
                throw new Error(data.error || 'Failed to load portfolio');
            }
            
        } catch (error) {
            console.error('Failed to load portfolio:', error);
        }
    }
    
    async loadSignals() {
        try {
            const response = await fetch('/api/signals?limit=10');
            const data = await response.json();
            
            if (response.ok) {
                this.updateSignals(data.signals);
            } else {
                throw new Error(data.error || 'Failed to load signals');
            }
            
        } catch (error) {
            console.error('Failed to load signals:', error);
        }
    }
    
    async loadPrices() {
        try {
            const response = await fetch('/api/prices');
            const data = await response.json();
            
            if (response.ok) {
                this.updatePrices(data.prices);
            } else {
                throw new Error(data.error || 'Failed to load prices');
            }
            
        } catch (error) {
            console.error('Failed to load prices:', error);
        }
    }
    
    async loadSymbolOptions() {
        try {
            // Populate symbol dropdown with supported pairs
            const symbolSelect = document.getElementById('symbol-select');
            const supportedPairs = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT', 
                                   'BNBUSDT', 'XRPUSDT', 'LTCUSDT', 'BCHUSDT', 'EOSUSDT'];
            
            symbolSelect.innerHTML = '<option value="">Select Symbol</option>';
            supportedPairs.forEach(symbol => {
                const option = document.createElement('option');
                option.value = symbol;
                option.textContent = symbol;
                symbolSelect.appendChild(option);
            });
            
        } catch (error) {
            console.error('Failed to load symbol options:', error);
        }
    }
    
    updateConnectionStatus(connected) {
        const statusElement = document.getElementById('system-status');
        if (connected) {
            statusElement.className = 'badge bg-success';
            statusElement.textContent = 'Connected';
        } else {
            statusElement.className = 'badge bg-danger';
            statusElement.textContent = 'Disconnected';
        }
    }
    
    updateSystemStatus(data) {
        // Update model info
        if (data.model) {
            const progress = data.model.training_progress || 0;
            document.getElementById('training-progress').style.width = progress + '%';
            document.getElementById('training-progress').textContent = progress + '%';
            
            const performance = data.model.model_performance || {};
            document.getElementById('model-accuracy').textContent = 
                performance.accuracy ? (performance.accuracy * 100).toFixed(2) + '%' : '--';
            
            document.getElementById('active-features').textContent = 
                data.model.selected_features_count || '--';
            
            if (performance.training_date) {
                const date = new Date(performance.training_date);
                document.getElementById('last-training').textContent = date.toLocaleString();
            }
        }
        
        // Update demo mode indicator
        const demoMode = data.demo_mode;
        const demoElement = document.getElementById('demo-mode');
        if (demoMode) {
            demoElement.className = 'badge bg-warning';
            demoElement.textContent = 'Demo Mode';
        } else {
            demoElement.className = 'badge bg-success';
            demoElement.textContent = 'Live Mode';
        }
        
        // Update system status
        const running = data.system_running;
        const statusElement = document.getElementById('system-status');
        if (running) {
            statusElement.className = 'badge bg-success';
            statusElement.textContent = 'Running';
        } else {
            statusElement.className = 'badge bg-warning';
            statusElement.textContent = 'Stopped';
        }
    }
    
    updateIndicators(data) {
        const activeElement = document.getElementById('active-indicators');
        const inactiveElement = document.getElementById('inactive-indicators');
        const activeCountElement = document.getElementById('active-indicators-count');
        const inactiveCountElement = document.getElementById('inactive-indicators-count');
        
        // Clear existing content
        activeElement.innerHTML = '';
        inactiveElement.innerHTML = '';
        
        // Separate indicators by status
        const activeIndicators = data.indicators.filter(ind => ind.status === 'active');
        const inactiveIndicators = data.indicators.filter(ind => ind.status === 'inactive');
        
        // Update counts
        activeCountElement.textContent = activeIndicators.length;
        inactiveCountElement.textContent = inactiveIndicators.length;
        
        // Populate active indicators
        if (activeIndicators.length > 0) {
            activeIndicators.forEach(indicator => {
                const div = document.createElement('div');
                div.className = 'indicator-item active';
                div.innerHTML = `
                    <span class="indicator-name">${indicator.name}</span>
                    <span class="indicator-importance">${(indicator.importance || 0).toFixed(3)}</span>
                `;
                activeElement.appendChild(div);
            });
        } else {
            activeElement.innerHTML = '<div class="text-muted">No active indicators</div>';
        }
        
        // Populate inactive indicators
        if (inactiveIndicators.length > 0) {
            inactiveIndicators.forEach(indicator => {
                const div = document.createElement('div');
                div.className = 'indicator-item inactive';
                div.innerHTML = `
                    <span class="indicator-name">${indicator.name}</span>
                    <span class="indicator-importance">${(indicator.importance || 0).toFixed(3)}</span>
                `;
                inactiveElement.appendChild(div);
            });
        } else {
            inactiveElement.innerHTML = '<div class="text-muted">No inactive indicators</div>';
        }
    }
    
    updatePortfolio(data) {
        document.getElementById('portfolio-balance').textContent = 
            '$' + (data.portfolio_balance || 0).toFixed(2);
        
        const totalPnl = data.total_pnl || 0;
        const totalPnlElement = document.getElementById('total-pnl');
        totalPnlElement.textContent = '$' + totalPnl.toFixed(2);
        totalPnlElement.className = totalPnl >= 0 ? 'h5 text-success' : 'h5 text-danger';
        
        const todayPnl = Object.values(data.daily_pnl || {}).pop()?.total || 0;
        const todayPnlElement = document.getElementById('today-pnl');
        todayPnlElement.textContent = '$' + todayPnl.toFixed(2);
        todayPnlElement.className = todayPnl >= 0 ? 'h5 text-success' : 'h5 text-danger';
        
        document.getElementById('active-positions').textContent = data.active_positions || 0;
        
        // Update positions list
        this.updatePositionsList(data.positions || []);
    }
    
    updatePositionsList(positions) {
        const positionsElement = document.getElementById('positions-list');
        
        if (positions.length === 0) {
            positionsElement.innerHTML = '<div class="text-center text-muted">No active positions</div>';
            return;
        }
        
        positionsElement.innerHTML = '';
        
        positions.forEach(position => {
            const div = document.createElement('div');
            div.className = `position-item ${position.side.toLowerCase()}`;
            
            const pnlClass = position.unrealized_pnl >= 0 ? 'positive' : 'negative';
            
            div.innerHTML = `
                <div class="position-header">
                    <span class="position-symbol">${position.symbol}</span>
                    <span class="position-side ${position.side.toLowerCase()}">${position.side}</span>
                </div>
                <div class="position-details">
                    <div class="position-detail">
                        <span>Entry:</span>
                        <span>$${position.entry_price.toFixed(4)}</span>
                    </div>
                    <div class="position-detail">
                        <span>Quantity:</span>
                        <span>${position.remaining_quantity.toFixed(6)}</span>
                    </div>
                    <div class="position-detail">
                        <span>TP Level:</span>
                        <span>${position.current_tp_level}/${position.total_tp_levels}</span>
                    </div>
                    <div class="position-detail">
                        <span>P&L:</span>
                        <span class="position-pnl ${pnlClass}">$${position.unrealized_pnl.toFixed(2)}</span>
                    </div>
                </div>
            `;
            
            positionsElement.appendChild(div);
        });
    }
    
    updateSignals(signals) {
        const tableBody = document.getElementById('signals-table');
        
        if (signals.length === 0) {
            tableBody.innerHTML = '<tr><td colspan="6" class="text-center text-muted">No recent signals</td></tr>';
            return;
        }
        
        tableBody.innerHTML = '';
        
        signals.forEach(signal => {
            const row = document.createElement('tr');
            
            const time = new Date(signal.timestamp).toLocaleTimeString();
            const signalClass = `signal-${signal.prediction.toLowerCase()}`;
            
            let confidenceClass = 'confidence-low';
            if (signal.confidence >= 80) confidenceClass = 'confidence-high';
            else if (signal.confidence >= 60) confidenceClass = 'confidence-medium';
            
            const statusClass = signal.executed ? 'status-executed' : 'status-pending';
            const statusText = signal.executed ? 'Executed' : 'Pending';
            
            row.innerHTML = `
                <td>${time}</td>
                <td>${signal.symbol}</td>
                <td class="${signalClass}">${signal.prediction}</td>
                <td class="${confidenceClass}">${signal.confidence.toFixed(1)}%</td>
                <td>$${signal.price.toFixed(4)}</td>
                <td><span class="badge ${statusClass}">${statusText}</span></td>
            `;
            
            tableBody.appendChild(row);
        });
    }
    
    updatePrices(prices) {
        const pricesElement = document.getElementById('current-prices');
        
        if (Object.keys(prices).length === 0) {
            pricesElement.innerHTML = '<div class="text-center text-muted">No price data available</div>';
            return;
        }
        
        pricesElement.innerHTML = '';
        
        Object.entries(prices).forEach(([symbol, data]) => {
            const div = document.createElement('div');
            div.className = 'price-item';
            div.innerHTML = `
                <span class="price-symbol">${symbol}</span>
                <span class="price-value">$${data.price.toFixed(4)}</span>
            `;
            pricesElement.appendChild(div);
        });
    }
    
    async loadTpSlConfig(symbol) {
        try {
            const response = await fetch(`/api/tp_sl_config/${symbol}`);
            const data = await response.json();
            
            if (response.ok) {
                document.getElementById('tp-levels').value = data.tp_levels.map(x => (x * 100).toFixed(1)).join(',');
                document.getElementById('sl-percentage').value = (data.sl_percentage * 100).toFixed(1);
                document.getElementById('max-positions').value = data.max_positions;
            }
            
        } catch (error) {
            console.error('Failed to load TP/SL config:', error);
            this.showAlert('Failed to load TP/SL configuration', 'danger');
        }
    }
    
    async saveTpSlConfig() {
        try {
            const symbol = document.getElementById('symbol-select').value;
            if (!symbol) {
                this.showAlert('Please select a symbol', 'warning');
                return;
            }
            
            const tpLevels = document.getElementById('tp-levels').value
                .split(',')
                .map(x => parseFloat(x.trim()) / 100)
                .filter(x => !isNaN(x));
            
            const slPercentage = parseFloat(document.getElementById('sl-percentage').value) / 100;
            const maxPositions = parseInt(document.getElementById('max-positions').value);
            
            if (tpLevels.length === 0 || isNaN(slPercentage) || isNaN(maxPositions)) {
                this.showAlert('Please fill all fields with valid values', 'warning');
                return;
            }
            
            const response = await fetch(`/api/tp_sl_config/${symbol}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    tp_levels: tpLevels,
                    sl_percentage: slPercentage,
                    max_positions: maxPositions
                })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                this.showAlert('TP/SL configuration saved successfully', 'success');
                // Close modal
                const modal = bootstrap.Modal.getInstance(document.getElementById('tpSlModal'));
                modal.hide();
            } else {
                throw new Error(data.error || 'Failed to save configuration');
            }
            
        } catch (error) {
            console.error('Failed to save TP/SL config:', error);
            this.showAlert('Failed to save TP/SL configuration', 'danger');
        }
    }
    
    async retrainModel() {
        try {
            const button = document.getElementById('retrain-btn');
            button.disabled = true;
            button.innerHTML = '<span class="loading"></span> Retraining...';
            
            const response = await fetch('/api/retrain_model', {
                method: 'POST'
            });
            
            const data = await response.json();
            
            if (response.ok) {
                this.showAlert('Model retraining started', 'success');
            } else {
                throw new Error(data.error || 'Failed to start retraining');
            }
            
        } catch (error) {
            console.error('Failed to retrain model:', error);
            this.showAlert('Failed to start model retraining', 'danger');
        } finally {
            const button = document.getElementById('retrain-btn');
            button.disabled = false;
            button.innerHTML = '<i class="fas fa-sync me-2"></i>Retrain Model';
        }
    }
    
    handleSystemUpdate(data) {
        // Handle real-time system updates via WebSocket
        this.updateSystemStatus(data);
        
        if (data.portfolio) {
            this.updatePortfolio(data.portfolio);
        }
        
        if (data.latest_signals) {
            this.updateSignals(data.latest_signals);
        }
        
        this.lastUpdate = new Date();
    }
    
    showAlert(message, type = 'info') {
        // Create and show bootstrap alert
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        alertDiv.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
        
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(alertDiv);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }
    
    showLoading(show) {
        // Could implement a loading overlay here
        console.log(show ? 'Loading...' : 'Loading complete');
    }
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new TradingDashboard();
});