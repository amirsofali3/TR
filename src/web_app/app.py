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
    
    @app.route('/api/retrain', methods=['POST'])
    def retrain_model():
        """Manually trigger model retraining (User Feedback Adjustments)"""
        try:
            if not app.trading_engine:
                return jsonify({'error': 'Trading engine not initialized'}), 500
            
            # Always use fast retrain per user feedback (no Option B)
            retrain_type = "fast"
            
            # Start retraining in background
            def retrain_background():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        app.trading_engine.manual_retrain(retrain_type)
                    )
                    # Store result for status polling if needed
                    app.last_retrain_result = result
                finally:
                    loop.close()
            
            thread = threading.Thread(target=retrain_background)
            thread.daemon = True
            thread.start()
            
            return jsonify({
                'started': True, 
                'mode': 'fast'
            }), 202
            
        except Exception as e:
            logger.error(f"Retrain model error: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/features')
    def get_features():
        """Get full selected and inactive features lists (User Feedback Adjustments)"""
        try:
            if not app.ml_model:
                return jsonify({'error': 'ML model not initialized'}), 500
            
            # Get model info for features
            model_info = app.ml_model.get_model_info()
            selected_features = model_info.get('active_features', [])
            inactive_features = model_info.get('inactive_features', [])
            
            # Get additional metadata
            feature_importance = model_info.get('feature_importance', {})
            
            # Build detailed feature information
            selected_detailed = []
            for feature in selected_features:
                selected_detailed.append({
                    'name': feature,
                    'importance': feature_importance.get(feature, 0.0),
                    'status': 'selected'
                })
            
            inactive_detailed = []
            for feature in inactive_features:
                inactive_detailed.append({
                    'name': feature,
                    'importance': feature_importance.get(feature, 0.0),
                    'status': 'inactive'
                })
            
            return jsonify({
                'selected': selected_detailed,
                'inactive': inactive_detailed,
                'metadata': {
                    'total_selected': len(selected_features),
                    'total_inactive': len(inactive_features),
                    'total_features': len(selected_features) + len(inactive_features),
                    'model_version': model_info.get('model_version', 1),
                    'last_training_time': model_info.get('last_training_time')
                }
            })
            
        except Exception as e:
            logger.error(f"Features API error: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/analysis/interval', methods=['POST'])
    def change_analysis_interval():
        """Change analysis interval dynamically (User Feedback Adjustments)"""
        try:
            if not app.trading_engine:
                return jsonify({'error': 'Trading engine not initialized'}), 500
            
            data = request.get_json()
            if not data or 'interval_sec' not in data:
                return jsonify({'error': 'interval_sec is required'}), 400
            
            interval_sec = data['interval_sec']
            
            # Validate interval (must be one of allowed values)
            allowed_intervals = [1, 5, 10, 15, 30, 60]
            if interval_sec not in allowed_intervals:
                return jsonify({
                    'error': f'Invalid interval. Allowed values: {allowed_intervals}'
                }), 400
            
            # Get current interval for logging
            current_status = app.trading_engine.get_system_status()
            old_interval = current_status.get('analysis', {}).get('interval_sec', BASE_ANALYSIS_INTERVAL_SEC)
            
            # Call trading engine method to change interval in background
            def change_interval_background():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    # For now, we'll store the new interval and let the trading engine pick it up
                    # This avoids complex async issues in the web route
                    app.trading_engine._pending_interval_change = interval_sec
                    
                    # Enhanced logging per user feedback
                    logger.info(f"[ANALYSIS] Interval change requested old={old_interval}s new={interval_sec}s reason=\"user_request\"")
                    
                    # Emit WebSocket event for interval change
                    socketio.emit('analysis_interval_changed', {
                        'old_interval_sec': old_interval,
                        'new_interval_sec': interval_sec,
                        'timestamp': datetime.now().isoformat()
                    })
                finally:
                    loop.close()
            
            thread = threading.Thread(target=change_interval_background)
            thread.daemon = True
            thread.start()
            
            return jsonify({
                'success': True,
                'old_interval_sec': old_interval,
                'new_interval_sec': interval_sec,
                'message': f'Analysis interval change requested from {old_interval}s to {interval_sec}s'
            })
            
        except Exception as e:
            logger.error(f"Change analysis interval error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/database/stats')
    def get_database_stats():
        """Get database statistics and diagnostics"""
        try:
            from src.database.db_manager import db_manager
            
            stats = {
                'backend': db_manager.backend,
                'backend_info': db_manager.get_backend_info(),
                'tables': {}
            }
            
            # Get table record counts
            tables = ['market_ticks', 'market_data', 'ohlc_1s', 'ohlc_1m', 
                     'model_training_runs', 'real_time_prices', 'positions']
            
            for table in tables:
                try:
                    result = db_manager.fetchone(f"SELECT COUNT(*) FROM {table}")
                    stats['tables'][table] = result[0] if result else 0
                except Exception as e:
                    stats['tables'][table] = f"Error: {str(e)}"
            
            # Get collection status if available
            if app.data_collector:
                collection_status = app.data_collector.get_bootstrap_status()
                stats['collection_status'] = collection_status
            
            return jsonify(stats)
            
        except Exception as e:
            logger.error(f"Database stats API error: {e}")
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
        """Send periodic updates to connected clients (Complete Pipeline Restructure)"""
        from src.utils.json_sanitize import sanitize_for_json
        
        last_collection_progress = None
        last_training_progress = None
        last_indicator_progress = None
        last_accuracy = None
        
        while True:
            try:
                if not app.trading_engine:
                    socketio.sleep(5)
                    continue
                    
                # Get comprehensive system status
                status = app.trading_engine.get_system_status()
                
                # Sanitize status for JSON serialization
                status = sanitize_for_json(status)
                
                # Emit general system update
                socketio.emit('system_update', status)
                
                # Emit specific events for Complete Pipeline Restructure + User Feedback
                
                # Collection progress event
                collection_status = status.get('collection', {})
                if collection_status != last_collection_progress:
                    socketio.emit('collection_progress', collection_status)
                    last_collection_progress = collection_status.copy() if collection_status else None
                
                # Indicator progress event (separate from training - User Feedback Adjustments)
                indicator_progress = status.get('indicator_progress', {})
                if indicator_progress != last_indicator_progress:
                    socketio.emit('indicator_progress', indicator_progress)
                    last_indicator_progress = indicator_progress.copy() if indicator_progress else None
                
                # Training progress event
                training_status = status.get('training', {})
                training_progress = training_status.get('progress_percent', 0)
                if training_progress != last_training_progress:
                    socketio.emit('training_progress', {
                        'progress_percent': training_progress,
                        'current_stage': training_status.get('phase'),
                        'training_info': training_status.get('training_progress_info', {})
                    })
                    last_training_progress = training_progress
                
                # Accuracy update event
                current_accuracy = training_status.get('accuracy_live', 0)
                if current_accuracy != last_accuracy:
                    socketio.emit('accuracy_update', {
                        'accuracy_live': current_accuracy,
                        'last_accuracy': training_status.get('last_accuracy', 0),
                        'accuracy_window_stats': training_status.get('accuracy_window_stats', {}),
                        'model_version': training_status.get('model_version', 1)
                    })
                    last_accuracy = current_accuracy
                
                # Status update event (comprehensive)
                socketio.emit('status_update', {
                    'timestamp': status.get('timestamp'),
                    'system_running': status.get('system_running', False),
                    'collection': collection_status,
                    'training': training_status,
                    'indicators': status.get('indicators', {}),
                    'backend': status.get('backend', {}),
                    'online_learning': status.get('online_learning', {})
                })
                
                socketio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Background update error: {e}")
                # Log the error details for debugging
                import traceback
                logger.debug(f"Background update error details: {traceback.format_exc()}")
                socketio.sleep(10)
    
    # Start background updates
    socketio.start_background_task(background_updates)
    
    return app