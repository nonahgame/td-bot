# Import dependencies
import os
import pandas as pd
import numpy as np
import sqlite3
import time
from datetime import datetime
import ccxt
import pandas_ta as ta
from telegram import Bot
import telegram
import logging
import threading
import requests
from flask import Flask, request, render_template, jsonify
import atexit
import subprocess

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler('at00_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)

# Environment variables
BOT_TOKEN = os.getenv("BOT_TOKEN", "6797387984:AAGQEK5Tdc-FNuJt3ecTEQ5eP6rrarDNMKA")  # Replace with your Telegram bot token
CHAT_ID = os.getenv("CHAT_ID", "672073574")        # Replace with your Telegram chat ID
SYMBOL = os.getenv("SYMBOL", "BTC/USD")
TIMEFRAME = os.getenv("TIMEFRAME", "5m")
STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", -0.15))
TAKE_PROFIT_PERCENT = float(os.getenv("TAKE_PROFIT_PERCENT", 2.0))
STOP_AFTER_SECONDS = float(os.getenv("STOP_AFTER_SECONDS", 43200))  # Stop after 1 hour; adjust as needed

# Global state
bot_thread = None
bot_active = True  # Start bot automatically
bot_lock = threading.Lock()
conn = None
exchange = ccxt.kraken()
position = None
buy_price = None
total_profit = 0
pause_duration = 0
pause_start = None
latest_signal = None
start_time = datetime.now()
stop_time = start_time + pd.Timedelta(seconds=STOP_AFTER_SECONDS)

# Keep-alive mechanism
def keep_alive():
    while True:
        try:
            requests.get('https://www.google.com')
            logger.debug("Keep-alive ping sent")
            time.sleep(300)
        except Exception as e:
            logger.error(f"Keep-alive error: {e}")
            time.sleep(60)

# SQLite database setup
def setup_database():
    global conn
    db_path = 'at00_bot.db'
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades';")
        if not c.fetchone():
            c.execute('''
                CREATE TABLE trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    time TEXT,
                    action TEXT,
                    symbol TEXT,
                    price REAL,
                    open_price REAL,
                    close_price REAL,
                    volume REAL,
                    percent_change REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    profit REAL,
                    total_profit REAL,
                    ema1 REAL,
                    ema2 REAL,
                    rsi REAL,
                    k REAL,
                    d REAL,
                    j REAL,
                    message TEXT,
                    timeframe TEXT
                )
            ''')
        c.execute("PRAGMA table_info(trades);")
        columns = [col[1] for col in c.fetchall()]
        for col in ['message', 'timeframe']:
            if col not in columns:
                c.execute(f'ALTER TABLE trades ADD COLUMN {col} TEXT;')
        conn.commit()
        logger.info(f"Database initialized at {db_path}")
    except Exception as e:
        logger.error(f"Database setup error: {e}")
        raise

# Fetch price data
def get_simulated_price(symbol=SYMBOL, exchange=exchange, timeframe=TIMEFRAME, retries=3, delay=5):
    for attempt in range(retries):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=1)
            if not ohlcv:
                logger.warning(f"No data returned for {symbol}. Retrying...")
                time.sleep(delay)
                continue
            data = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms', utc=True)
            logger.debug(f"Fetched price data: {data.iloc[-1].to_dict()}")
            return data.iloc[-1]
        except Exception as e:
            logger.error(f"Error fetching price (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay)
    logger.error(f"Failed to fetch price for {symbol} after {retries} attempts.")
    return pd.Series({'Open': np.nan, 'Close': np.nan, 'High': np.nan, 'Low': np.nan, 'Volume': np.nan})

# Calculate technical indicators
def add_technical_indicators(df):
    try:
        df['ema1'] = ta.ema(df['Close'], length=12)
        df['ema2'] = ta.ema(df['Close'], length=26)
        df['rsi'] = ta.rsi(df['Close'], length=14)
        kdj = ta.kdj(df['High'], df['Low'], df['Close'], length=9, signal=3)
        df['k'] = kdj['K_9_3']
        df['d'] = kdj['D_9_3']
        df['j'] = kdj['J_9_3']
        logger.debug(f"Technical indicators calculated: {df.iloc[-1][['ema1', 'ema2', 'rsi', 'k', 'd', 'j']].to_dict()}")
        return df
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return df

# AI decision logic
def ai_decision(df, stop_loss_percent=STOP_LOSS_PERCENT, take_profit_percent=TAKE_PROFIT_PERCENT, position=None, buy_price=None):
    if df.empty or len(df) < 1:
        logger.warning("DataFrame is empty or too small for decision.")
        return "hold", None, None
    latest = df.iloc[-1]
    close_price = latest['Close']
    open_price = latest['Open']
    stop_loss = None
    take_profit = None
    action = "hold"
    if position == "long" and buy_price is not None:
        stop_loss = buy_price * (1 + stop_loss_percent / 100)
        take_profit = buy_price * (1 + take_profit_percent / 100)
        if close_price <= stop_loss:
            logger.info("Stop-loss triggered.")
            action = "sell"
        elif close_price >= take_profit:
            logger.info("Take-profit triggered.")
            action = "sell"
    if action == "hold" and position is None and close_price > open_price:
        logger.info("Buy signal detected (Close > Open).")
        action = "buy"
    if action == "hold" and position == "long" and close_price < open_price:
        logger.info("Sell signal detected (Close < Open).")
        action = "sell"
    return action, stop_loss, take_profit

# Telegram message sending with retries
def send_telegram_message(signal, bot_token, chat_id, retries=3, delay=5):
    for attempt in range(retries):
        try:
            start_time = time.time()
            logger.debug(f"Attempt {attempt + 1}/{retries} to send Telegram message")
            bot = Bot(token=bot_token)
            message = f"""
Time: {signal['time']}
Timeframe: {signal['timeframe']}
Msg: {signal['message']}
Price: {signal['price']:.2f}
Open: {signal['open_price']:.2f}
Close: {signal['close_price']:.2f}
Volume: {signal['volume']:.2f}
% Change: {signal['percent_change']:.2f}%
EMA1 (12): {signal['ema1']:.2f}
EMA2 (26): {signal['ema2']:.2f}
RSI (14): {signal['rsi']:.2f}
KDJ K: {signal['k']:.2f}
KDJ D: {signal['d']:.2f}
KDJ J: {signal['j']:.2f}
{f"Stop-Loss: {signal['stop_loss']:.2f}" if signal['stop_loss'] is not None else ""}
{f"Take-Profit: {signal['take_profit']:.2f}" if signal['take_profit'] is not None else ""}
{f"Total Profit: {signal['total_profit']:.2f}" if signal['action'] in ["buy", "sell"] else ""}
{f"Profit: {signal['profit']:.2f}" if signal['action'] == "sell" else ""}
"""
            bot.send_message(chat_id=chat_id, text=message)
            elapsed = time.time() - start_time
            logger.info(f"Telegram message sent successfully in {elapsed:.2f} seconds")
            return
        except Exception as e:
            logger.error(f"Error sending Telegram message (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay)
    logger.error(f"Failed to send Telegram message after {retries} attempts")

# Trading bot logic
def trading_bot():
    global bot_active, position, buy_price, total_profit, pause_duration, pause_start, latest_signal, conn
    setup_database()
    bot = Bot(token=BOT_TOKEN)
    last_update_id = 0
    df = None

    # Fetch initial historical data
    for attempt in range(3):
        try:
            ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=100)
            if not ohlcv:
                logger.warning(f"No historical data for {SYMBOL}. Retrying...")
                time.sleep(5)
                continue
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            df['High'] = df['High'].fillna(df['Close'])
            df['Low'] = df['Low'].fillna(df['Close'])
            df = add_technical_indicators(df)
            logger.info(f"Initial df shape: {df.shape}")
            break
        except Exception as e:
            logger.error(f"Error fetching historical data (attempt {attempt + 1}/3): {e}")
            if attempt < 2:
                time.sleep(5)
            else:
                logger.error(f"Failed to fetch historical data for {SYMBOL}.")
                return

    while True:
        with bot_lock:
            # Check if stop time is reached
            if datetime.now() >= stop_time:
                bot_active = False
                if position == "long":
                    latest_data = get_simulated_price()
                    if not pd.isna(latest_data['Close']):
                        profit = latest_data['Close'] - buy_price
                        total_profit += profit
                        signal = create_signal("sell", latest_data['Close'], latest_data, df, profit, total_profit, "Bot stopped due to time limit")
                        store_signal(signal)
                        # Send shutdown message synchronously
                        send_telegram_message(signal, BOT_TOKEN, CHAT_ID)
                    position = None
                logger.info("Bot stopped due to time limit")
                break

            if not bot_active:
                time.sleep(10)
                continue

        try:
            # Handle pause
            if pause_start and pause_duration > 0:
                elapsed = (datetime.now() - pause_start).total_seconds()
                if elapsed < pause_duration:
                    logger.info(f"Bot paused, resuming in {int(pause_duration - elapsed)} seconds")
                    time.sleep(min(pause_duration - elapsed, 60))
                    continue
                else:
                    pause_start = None
                    pause_duration = 0
                    position = None
                    logger.info("Bot resumed after pause")

            # Fetch price data
            latest_data = get_simulated_price()
            if pd.isna(latest_data['Close']):
                logger.warning("Skipping cycle due to missing price data.")
                time.sleep(60)
                continue
            current_price = latest_data['Close']
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Process Telegram commands
            try:
                updates = bot.get_updates(offset=last_update_id, timeout=10)
                for update in updates:
                    if update.message and update.message.text:
                        text = update.message.text.strip()
                        command_chat_id = update.message.chat.id
                        if text == '/help':
                            bot.send_message(chat_id=command_chat_id, text="Commands: /help, /stop, /stopN, /start, /status, /performance, /count")
                        elif text == '/stop':
                            with bot_lock:
                                if bot_active and position == "long":
                                    profit = current_price - buy_price
                                    total_profit += profit
                                    signal = create_signal("sell", current_price, latest_data, df, profit, total_profit, "Bot stopped via Telegram")
                                    store_signal(signal)
                                    send_telegram_message(signal, BOT_TOKEN, CHAT_ID)
                                    position = None
                                bot_active = False
                            bot.send_message(chat_id=command_chat_id, text="Bot stopped.")
                        elif text.startswith('/stop') and text[5:].isdigit():
                            multiplier = int(text[5:])
                            timeframe_seconds = {'1m': 60, '5m': 300, '15m': 900, '30m': 1800, '1h': 3600, '1d': 86400}.get(TIMEFRAME, 300)
                            with bot_lock:
                                pause_duration = multiplier * timeframe_seconds
                                pause_start = datetime.now()
                                if position == "long":
                                    profit = current_price - buy_price
                                    total_profit += profit
                                    signal = create_signal("sell", current_price, latest_data, df, profit, total_profit, "Bot paused via Telegram")
                                    store_signal(signal)
                                    send_telegram_message(signal, BOT_TOKEN, CHAT_ID)
                                    position = None
                                bot_active = False
                            bot.send_message(chat_id=command_chat_id, text=f"Bot paused for {pause_duration/60} minutes.")
                        elif text == '/start':
                            with bot_lock:
                                if not bot_active:
                                    bot_active = True
                                    position = None
                                    pause_start = None
                                    pause_duration = 0
                                    bot.send_message(chat_id=command_chat_id, text="Bot started.")
                        elif text == '/status':
                            status = "active" if bot_active else f"paused for {int(pause_duration - (datetime.now() - pause_start).total_seconds())} seconds" if pause_start else "stopped"
                            bot.send_message(chat_id=command_chat_id, text=status)
                        elif text == '/performance':
                            bot.send_message(chat_id=command_chat_id, text=get_performance())
                        elif text == '/count':
                            bot.send_message(chat_id=command_chat_id, text=get_trade_counts())
                    last_update_id = update.update_id + 1
            except Exception as e:
                logger.error(f"Error processing Telegram updates: {e}")

            # Update dataframe
            new_row = pd.DataFrame({
                'Open': [latest_data['Open']],
                'Close': [latest_data['Close']],
                'High': [latest_data['High']],
                'Low': [latest_data['Low']],
                'Volume': [latest_data['Volume']]
            }, index=[pd.Timestamp.now(tz='UTC')])
            df = pd.concat([df, new_row]).tail(100)
            df = add_technical_indicators(df)

            # Generate signal
            prev_close = df['Close'].iloc[-2] if len(df) >= 2 else df['Close'].iloc[-1]
            percent_change = ((current_price - prev_close) / prev_close * 100) if prev_close != 0 else 0.0
            recommended_action, stop_loss, take_profit = ai_decision(df, position=position, buy_price=buy_price)

            with bot_lock:
                action = "hold"
                profit = 0
                msg = f"HOLD {SYMBOL} at {current_price:.2f}"
                if bot_active and recommended_action == "buy" and position is None:
                    position = "long"
                    buy_price = current_price
                    action = "buy"
                    msg = f"BUY {SYMBOL} at {current_price:.2f}"
                elif bot_active and recommended_action == "sell" and position == "long":
                    profit = current_price - buy_price
                    total_profit += profit
                    position = None
                    action = "sell"
                    msg = f"SELL {SYMBOL} at {current_price:.2f}, Profit: {profit:.2f}"
                    if stop_loss and current_price <= stop_loss:
                        msg += " (Stop-Loss)"
                    elif take_profit and current_price >= take_profit:
                        msg += " (Take-Profit)"

                signal = create_signal(action, current_price, latest_data, df, profit, total_profit, msg)
                store_signal(signal)
                if bot_active and action != "hold":
                    # Send Telegram message in a separate thread
                    threading.Thread(target=send_telegram_message, args=(signal, BOT_TOKEN, CHAT_ID), daemon=True).start()
                latest_signal = signal

            timeframe_seconds = {'1m': 60, '5m': 300, '15m': 900, '30m': 1800, '1h': 3600, '1d': 86400}.get(TIMEFRAME, 300)
            time.sleep(timeframe_seconds)
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
            time.sleep(60)

# Helper functions
def create_signal(action, current_price, latest_data, df, profit, total_profit, msg):
    latest = df.iloc[-1]
    return {
        'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'action': action,
        'symbol': SYMBOL,
        'price': current_price,
        'open_price': latest_data['Open'],
        'close_price': latest_data['Close'],
        'volume': latest_data['Volume'],
        'percent_change': ((current_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100) if len(df) >= 2 else 0.0,
        'stop_loss': None,
        'take_profit': None,
        'profit': profit,
        'timeframe': TIMEFRAME,
        'total_profit': total_profit,
        'ema1': latest['ema1'],
        'ema2': latest['ema2'],
        'rsi': latest['rsi'],
        'k': latest['k'],
        'd': latest['d'],
        'j': latest['j'],
        'message': msg
    }

def store_signal(signal):
    try:
        c = conn.cursor()
        c.execute('''
            INSERT INTO trades (
                time, action, symbol, price, open_price, close_price, volume,
                percent_change, stop_loss, take_profit, profit, total_profit,
                ema1, ema2, rsi, k, d, j, message, timeframe
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal['time'], signal['action'], signal['symbol'], signal['price'],
            signal['open_price'], signal['close_price'], signal['volume'],
            signal['percent_change'], signal['stop_loss'], signal['take_profit'],
            signal['profit'], signal['total_profit'],
            signal['ema1'], signal['ema2'], signal['rsi'],
            signal['k'], signal['d'], signal['j'], signal['message'], signal['timeframe']
        ))
        conn.commit()
        logger.debug("Signal stored successfully")
    except Exception as e:
        logger.error(f"Error storing signal: {e}")

def get_performance():
    try:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM trades")
        trade_count = c.fetchone()[0]
        logger.debug(f"Total trades in database: {trade_count}")
        if trade_count == 0:
            return "No trades available for performance analysis."

        c.execute("SELECT DISTINCT timeframe FROM trades")
        timeframes = [row[0] for row in c.fetchall()]
        message = "Performance Statistics by Timeframe:\n"
        for tf in timeframes:
            c.execute("SELECT MIN(time), MAX(time), SUM(profit), COUNT(*) FROM trades WHERE action='sell' AND profit IS NOT NULL AND timeframe=?", (tf,))
            result = c.fetchone()
            min_time, max_time, total_profit_db, win_trades = result if result else (None, None, None, 0)
            c.execute("SELECT COUNT(*) FROM trades WHERE action='sell' AND profit < 0 AND timeframe=?", (tf,))
            loss_trades = c.fetchone()[0]
            duration = (datetime.strptime(max_time, "%Y-%m-%d %H:%M:%S") - datetime.strptime(min_time, "%Y-%m-%d %H:%M:%S")).total_seconds() / 3600 if min_time and max_time else "N/A"
            total_profit_db = total_profit_db if total_profit_db is not None else 0
            message += f"""
Timeframe: {tf}
Duration (hours): {duration if duration != "N/A" else duration}
Win Trades: {win_trades}
Loss Trades: {loss_trades}
Total Profit: {total_profit_db:.2f}
"""
        logger.info("Performance data generated successfully")
        return message
    except Exception as e:
        logger.error(f"Error fetching performance: {e}")
        return f"Error fetching performance data: {str(e)}"

def get_trade_counts():
    try:
        c = conn.cursor()
        c.execute("SELECT DISTINCT timeframe FROM trades")
        timeframes = [row[0] for row in c.fetchall()]
        message = "Trade Counts by Timeframe:\n"
        for tf in timeframes:
            c.execute("SELECT COUNT(*), SUM(profit) FROM trades WHERE timeframe=?", (tf,))
            total_trades, total_profit_db = c.fetchone()
            c.execute("SELECT COUNT(*) FROM trades WHERE action='buy' AND timeframe=?", (tf,))
            buy_trades = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM trades WHERE action='sell' AND timeframe=?", (tf,))
            sell_trades = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM trades WHERE action='sell' AND profit > 0 AND timeframe=?", (tf,))
            win_trades = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM trades WHERE action='sell' AND profit < 0 AND timeframe=?", (tf,))
            loss_trades = c.fetchone()[0]
            total_profit_db = total_profit_db if total_profit_db is not None else 0
            message += f"""
Timeframe: {tf}
Total Trades: {total_trades}
Buy Trades: {buy_trades}
Sell Trades: {sell_trades}
Win Trades: {win_trades}
Loss Trades: {loss_trades}
Total Profit: {total_profit_db:.2f}
"""
        logger.debug("Trade counts generated successfully")
        return message
    except Exception as e:
        logger.error(f"Error fetching trade counts: {e}")
        return f"Error fetching trade counts: {str(e)}"

# Flask routes
@app.route('/')
def index():
    global latest_signal, stop_time
    status = "active" if bot_active else "stopped"
    try:
        c = conn.cursor()
        c.execute("SELECT * FROM trades ORDER BY time DESC LIMIT 16")
        trades = [dict(zip([col[0] for col in c.description], row)) for row in c.fetchall()]
        stop_time_str = stop_time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info("Rendering index.html with trades")
        return render_template('index.html', signal=latest_signal, status=status, timeframe=TIMEFRAME, trades=trades, stop_time=stop_time_str)
    except Exception as e:
        logger.error(f"Error rendering index.html: {e}")
        return jsonify({"error": "Failed to render template"}), 500

@app.route('/status')
def status():
    status = "active" if bot_active else "stopped"
    return jsonify({"status": status, "timeframe": TIMEFRAME, "stop_time": stop_time.strftime("%Y-%m-%d %H:%M:%S")})

@app.route('/performance')
def performance():
    return jsonify({"performance": get_performance()})

@app.route('/trades')
def trades():
    try:
        c = conn.cursor()
        c.execute("SELECT * FROM trades ORDER BY time DESC LIMIT 16")
        trades = [dict(zip([col[0] for col in c.description], row)) for row in c.fetchall()]
        logger.debug(f"Fetched {len(trades)} trades for /trades endpoint")
        return jsonify(trades)
    except Exception as e:
        logger.error(f"Error fetching trades: {e}")
        return jsonify({"error": "Failed to fetch trades"}), 500

# Cleanup
def cleanup():
    global conn
    if conn:
        conn.close()
        logger.info("Database connection closed")

atexit.register(cleanup)

# Start trading bot
if bot_thread is None or not bot_thread.is_alive():
    bot_thread = threading.Thread(target=trading_bot, daemon=True)
    bot_thread.start()
    logger.info("Trading bot started automatically")

# Run Flask app in Colab for testing
if __name__ == "__main__":
    # Verify template exists
    logger.info("Checking for templates/index.html")
    if not os.path.exists('templates/index.html'):
        logger.error("Template file 'templates/index.html' not found")
        raise FileNotFoundError("Template file 'templates/index.html' not found")
        
    # Start keep-alive thread
    logger.info("Starting keep-alive thread")
    keep_alive_thread = threading.Thread(target=keep_alive, daemon=True)
    keep_alive_thread.start()

    #
    
