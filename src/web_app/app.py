"""
Flask Web Application for Trading System Dashboard
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import asyncio
import threading
import json
from datetime import datetime
from typing import Dict, Any
from loguru import logger

from config.settings import *

def create_app(data_collector=None, indicator_engine=None, ml_model=None, trading_engine=None):
    """Create Flask application"""
    
    app = Flask(__name__)
    app.secret_key = SECRET_KEY
    
    # Enable CORS
    CORS(app, origins="*")
    
    # Initialize SocketIO
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
    
    # Store references to components
    app.data_collector = data_collector
    app.indicator_engine = indicator_engine
    app.ml_model = ml_model
    app.trading_engine = trading_engine
    
    @app.route('/')
    def dashboard():
        """Main dashboard"""
        return render_template('dashboard.html')
    
    @app.route('/api/status')
    def get_status():
        """Get system status"""
        try:
            if app.trading_engine:
                status = app.trading_engine.get_system_status()
            else:
                status = {'error': 'Trading engine not initialized'}
            
            return jsonify(status)
            
        except Exception as e:
            logger.error(f"Status API error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/indicators')
    def get_indicators():
        """Get indicator status"""
        try:
            if not app.indicator_engine or not app.ml_model:
                return jsonify({'error': 'Components not initialized'}), 500
            
            # Get model info to determine active/inactive indicators
            model_info = app.ml_model.get_model_info()
            active_features = model_info.get('active_features', [])
            inactive_features = model_info.get('inactive_features', [])
            feature_importance = model_info.get('feature_importance', {})
            
            # Get all indicators from config
            all_indicators = []
            for name, config in app.indicator_engine.indicators_config.items():
                status = 'active' if name in active_features else 'inactive'
                importance = feature_importance.get(name, 0.0)
                
                all_indicators.append({
                    'name': name,
                    'category': config['category'],
                    'status': status,
                    'rfe_eligible': config['rfe_eligible'],
                    'must_keep': config['must_keep'],
                    'importance': importance,
                    'parameters': config['parameters']
                })
            
            return jsonify({
                'indicators': all_indicators,
                'total_count': len(all_indicators),
                'active_count': len(active_features),
                'inactive_count': len(inactive_features)
            })
            
        except Exception as e:
            logger.error(f"Indicators API error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/portfolio')
    def get_portfolio():
        """Get portfolio information"""
        try:
            if not app.trading_engine or not app.trading_engine.risk_manager:
                return jsonify({'error': 'Risk manager not initialized'}), 500
            
            portfolio = app.trading_engine.risk_manager.get_portfolio_summary()
            return jsonify(portfolio)
            
        except Exception as e:
            logger.error(f"Portfolio API error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/signals')
    def get_signals():
        """Get latest trading signals"""
        try:
            if not app.trading_engine:
                return jsonify({'error': 'Trading engine not initialized'}), 500
            
            limit = request.args.get('limit', 20, type=int)
            signals = app.trading_engine.get_latest_signals(limit)
            
            return jsonify({'signals': signals})
            
        except Exception as e:
            logger.error(f"Signals API error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/prices')
    def get_prices():
        """Get current prices"""
        try:
            if not app.data_collector:
                return jsonify({'error': 'Data collector not initialized'}), 500
            
            prices = {}
            for symbol in SUPPORTED_PAIRS:
                # Use asyncio in a thread-safe way
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    price_data = loop.run_until_complete(
                        app.data_collector.get_real_time_price(symbol)
                    )
                    if price_data:
                        prices[symbol] = price_data
                finally:
                    loop.close()
            
            return jsonify({'prices': prices})
            
        except Exception as e:
            logger.error(f"Prices API error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/tp_sl_config/<symbol>')
    def get_tp_sl_config(symbol):
        """Get TP/SL configuration for a symbol"""
        try:
            if not app.trading_engine or not app.trading_engine.risk_manager:
                return jsonify({'error': 'Risk manager not initialized'}), 500
            
            config = app.trading_engine.risk_manager.get_tp_sl_config(symbol)
            
            return jsonify({
                'symbol': config.symbol,
                'tp_levels': config.tp_levels,
                'sl_percentage': config.sl_percentage,
                'trailing_step': config.trailing_step,
                'max_positions': config.max_positions
            })
            
        except Exception as e:
            logger.error(f"TP/SL config API error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/tp_sl_config/<symbol>', methods=['POST'])
    def update_tp_sl_config(symbol):
        """Update TP/SL configuration for a symbol"""
        try:
            if not app.trading_engine or not app.trading_engine.risk_manager:
                return jsonify({'error': 'Risk manager not initialized'}), 500
            
            data = request.get_json()
            
            # Import here to avoid circular imports
            from src.risk_management.risk_manager import TPSLConfig
            
            config = TPSLConfig(
                symbol=symbol,
                tp_levels=data.get('tp_levels', DEFAULT_TP_LEVELS),
                sl_percentage=data.get('sl_percentage', DEFAULT_SL_PERCENTAGE),
                trailing_step=data.get('trailing_step', TP_SL_TRAILING_STEP),
                max_positions=data.get('max_positions', 1)
            )
            
            app.trading_engine.risk_manager.update_tp_sl_config(symbol, config)
            
            return jsonify({'success': True, 'message': f'TP/SL config updated for {symbol}'})
            
        except Exception as e:
            logger.error(f"Update TP/SL config error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/close_position/<position_id>', methods=['POST'])
    def close_position(position_id):
        """Manually close a position"""
        try:
            if not app.trading_engine or not app.trading_engine.risk_manager:
                return jsonify({'error': 'Risk manager not initialized'}), 500
            
            data = request.get_json()
            price = data.get('price')
            
            if not price:
                return jsonify({'error': 'Price is required'}), 400
            
            # Use asyncio in a thread-safe way
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                success = loop.run_until_complete(
                    app.trading_engine.risk_manager.close_position(position_id, price, 'MANUAL')
                )
            finally:
                loop.close()
            
            if success:
                return jsonify({'success': True, 'message': 'Position closed successfully'})
            else:
                return jsonify({'error': 'Failed to close position'}), 500
            
        except Exception as e:
            logger.error(f"Close position error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/retrain_model', methods=['POST'])
    def retrain_model():
        """Manually trigger model retraining"""
        try:
            if not app.trading_engine:
                return jsonify({'error': 'Trading engine not initialized'}), 500
            
            # Start retraining in background
            def retrain_background():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(app.trading_engine.retrain_models())
                finally:
                    loop.close()
            
            thread = threading.Thread(target=retrain_background)
            thread.daemon = True
            thread.start()
            
            return jsonify({'success': True, 'message': 'Model retraining started'})
            
        except Exception as e:
            logger.error(f"Retrain model error: {e}")
            return jsonify({'error': str(e)}), 500
    
    # SocketIO events for real-time updates
    @socketio.on('connect')
    def handle_connect():
        """Handle client connection"""
        logger.info("Client connected to WebSocket")
        emit('status', {'message': 'Connected to trading system'})
    
    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection"""
        logger.info("Client disconnected from WebSocket")
    
    @socketio.on('request_update')
    def handle_update_request():
        """Handle request for system update"""
        try:
            if app.trading_engine:
                status = app.trading_engine.get_system_status()
                emit('system_update', status)
            
        except Exception as e:
            logger.error(f"WebSocket update error: {e}")
            emit('error', {'message': str(e)})
    
    # Background task to send periodic updates
    def background_updates():
        """Send periodic updates to connected clients"""
        while True:
            try:
                if app.trading_engine:
                    status = app.trading_engine.get_system_status()
                    socketio.emit('system_update', status)
                
                socketio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Background update error: {e}")
                socketio.sleep(10)
    
    # Start background updates
    socketio.start_background_task(background_updates)
    
    return app