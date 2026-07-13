#!/usr/bin/env python3
"""
CLI tool for AI Crypto Trading System
Provides easy command-line interface for common operations
"""

import click
import sys
import os
from pathlib import Path
import subprocess
import json
import pandas as pd
from datetime import datetime, timedelta
from tabulate import tabulate
import requests
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

# ANSI color codes
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_success(message):
    click.echo(f"{Colors.GREEN}✓ {message}{Colors.ENDC}")

def print_error(message):
    click.echo(f"{Colors.RED}✗ {message}{Colors.ENDC}")

def print_warning(message):
    click.echo(f"{Colors.YELLOW}⚠ {message}{Colors.ENDC}")

def print_info(message):
    click.echo(f"{Colors.BLUE}ℹ {message}{Colors.ENDC}")

@click.group()
@click.version_option(version='1.0.0')
def cli():
    """AI Crypto Trading System CLI - Manage your trading bot with ease"""
    pass

@cli.command()
@click.option('--full', is_flag=True, help='Run full system with all services')
@click.option('--mode', type=click.Choice(['live', 'paper', 'backtest']), default='paper', help='Trading mode')
def start(full, mode):
    """Start the trading system"""
    click.echo(f"{Colors.BOLD}Starting AI Crypto Trading System...{Colors.ENDC}")
    
    if full:
        print_info("Starting full system with Docker Compose...")
        try:
            subprocess.run(['docker-compose', 'up', '-d'], check=True)
            print_success("All services started successfully!")
            click.echo("\nServices running:")
            click.echo("  📊 Dashboard: http://localhost:8501")
            click.echo("  📈 Grafana: http://localhost:3000")
            click.echo("  🔍 Prometheus: http://localhost:9090")
        except subprocess.CalledProcessError:
            print_error("Failed to start services. Is Docker running?")
    else:
        print_info(f"Starting trading bot in {mode} mode...")
        env = os.environ.copy()
        
        if mode == 'live':
            print_warning("Starting LIVE TRADING mode. Real money at risk!")
            if not click.confirm("Are you sure you want to start live trading?"):
                return
            env['ENABLE_LIVE_TRADING'] = 'true'
            env['ENABLE_PAPER_TRADING'] = 'false'
        elif mode == 'paper':
            env['ENABLE_LIVE_TRADING'] = 'false'
            env['ENABLE_PAPER_TRADING'] = 'true'
        
        try:
            subprocess.Popen([sys.executable, 'src/main.py'], env=env)
            print_success(f"Trading bot started in {mode} mode!")
        except Exception as e:
            print_error(f"Failed to start trading bot: {e}")

@cli.command()
def stop():
    """Stop the trading system"""
    print_info("Stopping trading system...")
    try:
        subprocess.run(['docker-compose', 'down'], check=True)
        print_success("All services stopped successfully!")
    except subprocess.CalledProcessError:
        print_warning("Docker services not running or already stopped")
    
    # Also try to stop any Python processes
    try:
        subprocess.run(['pkill', '-f', 'src/main.py'], check=False)
        subprocess.run(['pkill', '-f', 'streamlit'], check=False)
    except:
        pass

@cli.command()
@click.option('--service', type=click.Choice(['all', 'trading-bot', 'dashboard', 'sentiment']), default='all')
@click.option('--tail', default=100, help='Number of lines to show')
def logs(service, tail):
    """View system logs"""
    if service == 'all':
        # Check if using Docker
        try:
            subprocess.run(['docker-compose', 'logs', '--tail', str(tail), '-f'], check=True)
        except:
            # Fallback to local logs
            log_file = f"logs/trading_{datetime.now().strftime('%Y%m%d')}.log"
            if os.path.exists(log_file):
                subprocess.run(['tail', '-f', '-n', str(tail), log_file])
            else:
                print_warning("No logs found")
    else:
        try:
            subprocess.run(['docker-compose', 'logs', '--tail', str(tail), '-f', service], check=True)
        except:
            print_error(f"Could not get logs for {service}")

@cli.command()
def status():
    """Check system status"""
    click.echo(f"{Colors.BOLD}System Status{Colors.ENDC}\n")
    
    # Check Docker services
    try:
        result = subprocess.run(['docker-compose', 'ps'], capture_output=True, text=True)
        if result.returncode == 0:
            click.echo("Docker Services:")
            click.echo(result.stdout)
    except:
        print_warning("Docker not available")
    
    # Check health endpoint
    try:
        response = requests.get('http://localhost:8080/health', timeout=2)
        if response.status_code == 200:
            health_data = response.json()
            print_success(f"Trading Bot: {health_data.get('status', 'unknown')}")
        else:
            print_error("Trading Bot: unhealthy")
    except:
        print_error("Trading Bot: not running")
    
    # Check dashboard
    try:
        response = requests.get('http://localhost:8501', timeout=2)
        if response.status_code == 200:
            print_success("Dashboard: running")
        else:
            print_error("Dashboard: not accessible")
    except:
        print_error("Dashboard: not running")
    
    # Check Redis
    try:
        import redis
        r = redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'))
        r.ping()
        print_success("Redis: connected")
    except:
        print_error("Redis: not connected")

@cli.command()
def positions():
    """View current trading positions"""
    click.echo(f"{Colors.BOLD}Active Positions{Colors.ENDC}\n")
    
    try:
        # Try to get from database
        from src.utils.database import DatabaseManager
        db = DatabaseManager(os.getenv('DB_PATH', './data/trading_system.db'))
        positions_df = db.get_active_positions()
        
        if not positions_df.empty:
            # Format for display
            display_df = positions_df[['symbol', 'side', 'quantity', 'entry_price', 'stop_loss', 'take_profit']]
            click.echo(tabulate(display_df, headers='keys', tablefmt='grid', showindex=False))
            
            # Calculate total exposure
            total_value = (positions_df['quantity'] * positions_df['entry_price']).sum()
            click.echo(f"\nTotal Position Value: ${total_value:,.2f}")
        else:
            print_info("No active positions")
            
    except Exception as e:
        print_error(f"Could not fetch positions: {e}")

@cli.command()
@click.option('--period', type=click.Choice(['today', 'week', 'month', 'all']), default='today')
def performance(period):
    """View trading performance"""
    click.echo(f"{Colors.BOLD}Performance Summary - {period.capitalize()}{Colors.ENDC}\n")
    
    try:
        from src.utils.database import DatabaseManager
        from src.utils.metrics import MetricsCalculator
        
        db = DatabaseManager(os.getenv('DB_PATH', './data/trading_system.db'))
        
        # Determine date range
        end_date = datetime.now()
        if period == 'today':
            start_date = end_date.replace(hour=0, minute=0, second=0)
        elif period == 'week':
            start_date = end_date - timedelta(days=7)
        elif period == 'month':
            start_date = end_date - timedelta(days=30)
        else:
            start_date = None
        
        # Get metrics
        metrics = db.get_performance_metrics(start_date, end_date)
        
        # Display metrics
        click.echo(f"Total Trades: {metrics.get('total_trades', 0)}")
        click.echo(f"Win Rate: {metrics.get('win_rate', 0):.1%}")
        click.echo(f"Total Return: {metrics.get('total_return', 0):.2%}")
        click.echo(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        click.echo(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        click.echo(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        
        # Show recent trades
        click.echo(f"\n{Colors.BOLD}Recent Trades:{Colors.ENDC}")
        trades_df = db.get_recent_trades(limit=5)
        if not trades_df.empty:
            display_df = trades_df[['timestamp', 'symbol', 'side', 'pnl', 'status']]
            display_df['pnl'] = display_df['pnl'].apply(lambda x: f"{x:.2%}")
            click.echo(tabulate(display_df, headers='keys', tablefmt='grid', showindex=False))
        
    except Exception as e:
        print_error(f"Could not fetch performance data: {e}")

@cli.command()
@click.argument('strategy')
@click.option('--symbol', default='BTC/USDT', help='Trading symbol')
@click.option('--start-date', default=None, help='Start date (YYYY-MM-DD)')
@click.option('--end-date', default=None, help='End date (YYYY-MM-DD)')
@click.option('--optimize', is_flag=True, help='Run parameter optimization')
def backtest(strategy, symbol, start_date, end_date, optimize):
    """Run a backtest"""
    click.echo(f"{Colors.BOLD}Running Backtest{Colors.ENDC}")
    click.echo(f"Strategy: {strategy}")
    click.echo(f"Symbol: {symbol}")
    
    cmd = [sys.executable, 'src/backtest.py']
    
    if optimize:
        cmd.extend(['optimize', '--strategy', strategy])
    else:
        cmd.extend(['single', '--strategy', strategy])
    
    cmd.extend(['--symbol', symbol])
    
    if start_date:
        cmd.extend(['--start-date', start_date])
    if end_date:
        cmd.extend(['--end-date', end_date])
    
    try:
        subprocess.run(cmd, check=True)
        print_success("Backtest completed! Check reports/backtest_report.html")
    except subprocess.CalledProcessError:
        print_error("Backtest failed")

@cli.command()
def strategies():
    """List available trading strategies"""
    click.echo(f"{Colors.BOLD}Available Trading Strategies{Colors.ENDC}\n")
    
    strategies = [
        {
            'name': 'ma_crossover',
            'description': 'Moving Average Crossover',
            'params': 'fast_period, slow_period'
        },
        {
            'name': 'rsi_mean_reversion',
            'description': 'RSI Mean Reversion',
            'params': 'period, oversold, overbought'
        },
        {
            'name': 'breakout',
            'description': 'Price Breakout',
            'params': 'lookback'
        },
        {
            'name': 'ml_ensemble',
            'description': 'Machine Learning Ensemble',
            'params': 'confidence_threshold'
        },
        {
            'name': 'pattern_based',
            'description': 'Chart Pattern Recognition',
            'params': 'min_confidence'
        }
    ]
    
    for strategy in strategies:
        click.echo(f"{Colors.GREEN}{strategy['name']}{Colors.ENDC}")
        click.echo(f"  Description: {strategy['description']}")
        click.echo(f"  Parameters: {strategy['params']}\n")

@cli.command()
def dashboard():
    """Open the web dashboard"""
    print_info("Starting dashboard...")
    try:
        subprocess.Popen(['streamlit', 'run', 'src/dashboard/app.py'])
        print_success("Dashboard started! Opening in browser...")
        
        # Try to open in browser
        import webbrowser
        webbrowser.open('http://localhost:8501')
    except Exception as e:
        print_error(f"Failed to start dashboard: {e}")

@cli.command()
@click.confirmation_option(prompt='Are you sure you want to close all positions?')
def close_all():
    """Close all trading positions"""
    print_warning("Closing all positions...")
    
    try:
        import redis
        r = redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'))
        r.publish('system:commands', json.dumps({
            'command': 'close_all_positions',
            'timestamp': datetime.now().isoformat()
        }))
        print_success("Close all positions command sent")
    except Exception as e:
        print_error(f"Failed to send command: {e}")

@cli.command()
def config():
    """View current configuration"""
    click.echo(f"{Colors.BOLD}Current Configuration{Colors.ENDC}\n")
    
    # Display key configuration values
    config_items = [
        ('Trading Mode', 'ENABLE_LIVE_TRADING'),
        ('Initial Capital', 'INITIAL_CAPITAL'),
        ('Max Position Size', 'MAX_POSITION_SIZE'),
        ('Risk Per Trade', 'RISK_PER_TRADE'),
        ('Max Drawdown Limit', 'MAX_DRAWDOWN_LIMIT'),
        ('Trading Symbols', 'TRADING_SYMBOLS'),
        ('Sentiment Analysis', 'ENABLE_SENTIMENT_ANALYSIS'),
        ('Pattern Discovery', 'ENABLE_PATTERN_DISCOVERY'),
        ('Auto Retrain', 'ENABLE_AUTO_RETRAIN')
    ]
    
    for name, env_var in config_items:
        value = os.getenv(env_var, 'Not set')
        if env_var == 'ENABLE_LIVE_TRADING' and value == 'true':
            click.echo(f"{name}: {Colors.RED}{value} (LIVE TRADING){Colors.ENDC}")
        else:
            click.echo(f"{name}: {value}")

@cli.command()
def health():
    """Run system health check"""
    click.echo(f"{Colors.BOLD}System Health Check{Colors.ENDC}\n")
    
    checks = []
    
    # Check Python version
    python_version = sys.version.split()[0]
    if sys.version_info >= (3, 10):
        checks.append(('Python Version', python_version, 'OK'))
    else:
        checks.append(('Python Version', python_version, 'FAIL'))
    
    # Check required directories
    for dir_name in ['data', 'logs', 'models', 'config']:
        if os.path.exists(dir_name):
            checks.append((f'{dir_name}/ directory', 'exists', 'OK'))
        else:
            checks.append((f'{dir_name}/ directory', 'missing', 'FAIL'))
    
    # Check .env file
    if os.path.exists('.env'):
        checks.append(('.env file', 'exists', 'OK'))
    else:
        checks.append(('.env file', 'missing', 'FAIL'))
    
    # Check database
    db_path = os.getenv('DB_PATH', './data/trading_system.db')
    if os.path.exists(db_path):
        checks.append(('Database', 'exists', 'OK'))
    else:
        checks.append(('Database', 'missing', 'FAIL'))
    
    # Display results
    for check, value, status in checks:
        if status == 'OK':
            click.echo(f"{Colors.GREEN}✓{Colors.ENDC} {check}: {value}")
        else:
            click.echo(f"{Colors.RED}✗{Colors.ENDC} {check}: {value}")
    
    # Overall status
    if all(status == 'OK' for _, _, status in checks):
        click.echo(f"\n{Colors.GREEN}All checks passed!{Colors.ENDC}")
    else:
        click.echo(f"\n{Colors.YELLOW}Some checks failed. Run setup.sh to fix issues.{Colors.ENDC}")

if __name__ == '__main__':
    cli()
