# app.py
import os
import pandas as pd
import numpy as np
import sqlite3
import time
from datetime import datetime, timedelta
import ccxt
import pandas_ta as ta
from telegram import Bot
import telegram
import logging
import threading
import requests
from flask import Flask, render_template, jsonify
import atexit
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import io
import json

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler('renda_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Try to import dotenv, with fallback if not installed
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.debug("Loaded environment variables from .env file")
except ImportError:
    logger.warning("python-dotenv not installed. Relying on system environment variables.")

# Flask app setup
app = Flask(__name__)

# Environment variables
BOT_TOKEN = os.getenv("BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
CHAT_ID = os.getenv("CHAT_ID", "YOUR_CHAT_ID_HERE")
SYMBOL = os.getenv("SYMBOL", "BTC/USD")
TIMEFRAME = os.getenv("TIMEFRAME", "TIMEFRAME")
STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", -0.15))
TAKE_PROFIT_PERCENT = float(os.getenv("TAKE_PROFIT_PERCENT", 2.0))
STOP_AFTER_SECONDS = float(os.getenv("STOP_AFTER_SECONDS", 43200))
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "YOUR_FOLDER_ID")

# Google Drive setup
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def authenticate_google_drive():
    try:
        service_account_info = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
        if not service_account_info:
            logger.error("GOOGLE_SERVICE_ACCOUNT_JSON environment variable is not set.")
            return None
        try:
            creds_info = json.loads(service_account_info)
            logger.debug(f"Service account JSON keys: {list(creds_info.keys())}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse GOOGLE_SERVICE_ACCOUNT_JSON: {e}")
            return None
        creds = Credentials.from_service_account_info(creds_info, scopes=SCOPES)
        service = build('drive', 'v3', credentials=creds)
        logger.debug("Google Drive service initialized")
        return service
    except Exception as e:
        logger.error(f"Unexpected error in Google Drive authentication: {e}", exc_info=True)
        return None

def create_folder_if_not_exists(drive_service, folder_id, folder_name="RendaBotBackups"):
    try:
        folder = drive_service.files().get(fileId=folder_id, fields='id, name, mimeType').execute()
        if folder.get('mimeType') != 'application/vnd.google-apps.folder':
            logger.error(f"Provided folder ID {folder_id} is not a folder. MIME type: {folder.get('mimeType')}")
            return None
        logger.debug(f"Validated folder: {folder.get('name')} (ID: {folder_id})")
        return folder_id
    except Exception as e:
        if "File not found" in str(e):
            logger.warning(f"Folder ID {folder_id} not found. Creating new folder: {folder_name}")
            try:
                file_metadata = {
                    'name': folder_name,
                    'mimeType': 'application/vnd.google-apps.folder'
                }
                folder = drive_service.files().create(body=file_metadata, fields='id, name').execute()
                new_folder_id = folder.get('id')
                logger.info(f"Created new folder: {folder.get('name')} (ID: {new_folder_id})")
                logger.warning(f"Please update GOOGLE_DRIVE_FOLDER_ID in Render to: {new_folder_id}")
                return new_folder_id
            except Exception as create_e:
                logger.error(f"Failed to create folder {folder_name}: {create_e}", exc_info=True)
                return None
        else:
            logger.error(f"Error validating folder ID {folder_id}: {e}", exc_info=True)
            return None

def upload_to_google_drive(file_path, file_name, folder_id):
    try:
        drive_service = authenticate_google_drive()
        if not drive_service:
            logger.warning("Google Drive service not available. Skipping upload.")
            return
        if not folder_id or folder_id == "YOUR_FOLDER_ID":
            logger.error("GOOGLE_DRIVE_FOLDER_ID is not set or invalid.")
            return
        folder_id = create_folder_if_not_exists(drive_service, folder_id)
        if not folder_id:
            logger.error(f"Skipping upload due to invalid or inaccessible folder ID: {folder_id}")
            return
        logger.debug(f"Uploading {file_name} to folder ID: {folder_id}")
        query = f"name='{file_name}' and '{folder_id}' in parents and trashed=false"
        response = drive_service.files().list(q=query, fields='files(id, name)').execute()
        files = response.get('files', [])
        file_metadata = {
            'name': file_name,
            'parents': [folder_id]
        }
        media = MediaFileUpload(file_path)
        if files:
            file_id = files[0]['id']
            file = drive_service.files().update(
                fileId=file_id,
                media_body=media,
                fields='id'
            ).execute()
            logger.info(f"Updated {file_name} on Google Drive with ID: {file.get('id')}")
        else:
            file = drive_service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            logger.info(f"Uploaded {file_name} to Google Drive with ID: {file.get('id')}")
    except Exception as e:
        logger.error(f"Error uploading {file_name} to Google Drive: {e}", exc_info=True)

def download_from_google_drive(file_name, folder_id, destination_path):
    try:
        drive_service = authenticate_google_drive()
        if not drive_service:
            logger.warning("Google Drive service not available. Starting with a new database.")
            return False
        if not folder_id or folder_id == "YOUR_FOLDER_ID":
            logger.error("GOOGLE_DRIVE_FOLDER_ID is not set or invalid.")
            return False
        folder_id = create_folder_if_not_exists(drive_service, folder_id)
        if not folder_id:
            logger.error(f"Skipping download due to invalid or inaccessible folder ID: {folder_id}")
            return False
        logger.debug(f"Downloading {file_name} from folder ID: {folder_id}")
        query = f"name='{file_name}' and '{folder_id}' in parents and trashed=false"
        response = drive_service.files().list(q=query, fields='files(id, name)').execute()
        files = response.get('files', [])
        if not files:
            logger.info(f"No {file_name} found in Google Drive. Starting with a new database.")
            return False
        file_id = files[0]['id']
        request = drive_service.files().get_media(fileId=file_id)
        fh = io.FileIO(destination_path, 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            logger.debug(f"Download {file_name}: {int(status.progress() * 100)}%")
        logger.info(f"Downloaded {file_name} from Google Drive to {destination_path}")
        return True
    except Exception as e:
        logger.error(f"Error downloading {file_name} from Google Drive: {e}", exc_info=True)
        return False

# Global state
bot_thread = None
bot_active = True
bot_lock = threading.Lock()
conn = None
exchange = ccxt.kraken()
position = None
buy_price = None
total_profit = 0
pause_duration = 0
pause_start = None
latest_signal = {
    'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'action': 'hold',
    'symbol': SYMBOL,
    'price': 0.0,
    'open_price': 0.0,
    'close_price': 0.0,
    'volume': 0.0,
    'percent_change': 0.0,
    'stop_loss': None,
    'take_profit': None,
    'profit': 0.0,
    'total_profit': 0.0,
    'ema1': 0.0,
    'ema2': 0.0,
    'rsi': 0.0,
    'diff': 0.0,
    'k': 0.0,
    'd': 0.0,
    'j': 0.0,
    'message': 'Initializing...',
    'timeframe': TIMEFRAME
}
start_time = datetime.now()
stop_time = start_time + pd.Timedelta(seconds=STOP_AFTER_SECONDS)
last_valid_price = None

# SQLite database setup
def setup_database():
    global conn
    db_path = 'renda_bot.db'
    try:
        if download_from_google_drive('renda_bot.db', GOOGLE_DRIVE_FOLDER_ID, db_path):
            logger.info(f"Restored database from Google Drive to {db_path}")
        else:
            logger.info(f"No existing database found. Creating new database at {db_path}")
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
                    diff REAL,
                    message TEXT,
                    timeframe TEXT
                )
            ''')
        c.execute("PRAGMA table_info(trades);")
        columns = [col[1] for col in c.fetchall()]
        for col in ['message', 'timeframe', 'diff']:
            if col not in columns:
                c.execute(f'ALTER TABLE trades ADD COLUMN {col} {"TEXT" if col != "diff" else "REAL"};')
        conn.commit()
        logger.info(f"Database initialized at {db_path}")
    except Exception as e:
        logger.error(f"Database setup error: {e}")
        conn = None

# Initialize database immediately
logger.info("Initializing database at startup")
setup_database()
if conn is None:
    logger.error("Failed to initialize database. Flask routes may fail.")

# Fetch latest signal from database
def get_latest_signal_from_db():
    try:
        if conn is None:
            logger.error("Cannot fetch latest signal: Database connection is None")
            return None
        c = conn.cursor()
        c.execute("SELECT * FROM trades ORDER BY time DESC LIMIT 1")
        row = c.fetchone()
        if row:
            columns = [col[0] for col in c.description]
            signal = dict(zip(columns, row))
            logger.debug(f"Fetched latest signal from DB: {signal}")
            return signal
        return None
    except Exception as e:
        logger.error(f"Error fetching latest signal from DB: {e}")
        return None

# Delete Telegram webhook with retries
def delete_webhook(retries=3, delay=5):
    try:
        bot = Bot(token=BOT_TOKEN)
        for attempt in range(retries):
            try:
                bot.delete_webhook()
                logger.info("Telegram webhook deleted successfully")
                return True
            except telegram.error.TelegramError as e:
                logger.error(f"Failed to delete webhook (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(delay)
        logger.error(f"Failed to delete webhook after {retries} attempts")
        return False
    except Exception as e:
        logger.error(f"Error initializing bot for webhook deletion: {e}")
        return False

# Fetch price data with improved handling
def get_simulated_price(symbol=SYMBOL, exchange=exchange, timeframe=TIMEFRAME, retries=3, delay=5):
    global last_valid_price
    for attempt in range(retries):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=5)
            if not ohlcv:
                logger.warning(f"No data returned for {symbol}. Retrying...")
                time.sleep(delay)
                continue
            data = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms', utc=True)
            data['diff'] = data['Close'] - data['Open']
            non_zero_diff = data[abs(data['diff']) > 0]
            selected_data = non_zero_diff.iloc[-1] if not non_zero_diff.empty else data.iloc[-1]
            if abs(selected_data['diff']) < 0.01:
                logger.warning(f"Open and Close similar for {symbol} (diff={selected_data['diff']}). Accepting data.")
            last_valid_price = selected_data
            logger.debug(f"Fetched price data: {selected_data.to_dict()}")
            return selected_data
        except Exception as e:
            logger.error(f"Error fetching price (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay)
    logger.error(f"Failed to fetch price for {symbol} after {retries} attempts.")
    if last_valid_price is not None:
        logger.info("Using last valid price data as fallback.")
        return last_valid_price
    return pd.Series({'Open': np.nan, 'Close': np.nan, 'High': np.nan, 'Low': np.nan, 'Volume': np.nan, 'diff': np.nan})

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
        df['diff'] = df['Close'] - df['Open']
        logger.debug(f"Technical indicators calculated: {df.iloc[-1][['ema1', 'ema2', 'rsi', 'k', 'd', 'j', 'diff']].to_dict()}")
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
        elif close_price < open_price:
            logger.info("Sell signal detected (Close < Open).")
            action = "sell"

    if action == "hold" and position is None and close_price > open_price:
        logger.info("Buy signal detected (Close > Open).")
        action = "buy"

    if action == "buy" and position is not None:
        logger.debug("Prevented consecutive buy order.")
        action = "hold"
    if action == "sell" and position is None:
        logger.debug("Prevented sell order without open position.")
        action = "hold"

    logger.debug(f"AI decision: action={action}, stop_loss={stop_loss}, take_profit={take_profit}")
    return action, stop_loss, take_profit

# Telegram message sending
def send_telegram_message(signal, bot_token, chat_id, retries=3, delay=5):
    for attempt in range(retries):
        try:
            bot = Bot(token=bot_token)
            diff_color = "🟢" if signal['diff'] > 0 else "🔴"
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
Diff: {diff_color} {signal['diff']:.2f}
KDJ K: {signal['k']:.2f}
KDJ D: {signal['d']:.2f}
KDJ J: {signal['j']:.2f}
{f"Stop-Loss: {signal['stop_loss']:.2f}" if signal['stop_loss'] is not None else ""}
{f"Take-Profit: {signal['take_profit']:.2f}" if signal['take_profit'] is not None else ""}
{f"Total Profit: {signal['total_profit']:.2f}" if signal['action'] in ["buy", "sell"] else ""}
{f"Profit: {signal['profit']:.2f}" if signal['action'] == "sell" else ""}
"""
            bot.send_message(chat_id=chat_id, text=message)
            logger.info(f"Telegram message sent successfully")
            return
        except Exception as e:
            logger.error(f"Error sending Telegram message (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay)
    logger.error(f"Failed to send Telegram message after {retries} attempts")

# Parse timeframe to seconds
def timeframe_to_seconds(timeframe):
    try:
        value = int(timeframe[:-1])
        unit = timeframe[-1].lower()
        if unit == 'm':
            return value * 60
        elif unit == 'h':
            return value * 3600
        elif unit == 'd':
            return value * 86400
        else:
            logger.error(f"Unsupported timeframe unit: {unit}")
            return 60
    except Exception as e:
        logger.error(f"Error parsing timeframe {timeframe}: {e}")
        return 60

# Align to next time boundary
def align_to_next_boundary(interval_seconds):
    now = datetime.now()
    seconds_since_epoch = now.timestamp()
    seconds_to_next = interval_seconds - (seconds_since_epoch % interval_seconds)
    if seconds_to_next == interval_seconds:
        seconds_to_next = 0
    next_boundary = now + timedelta(seconds=seconds_to_next)
    next_boundary = next_boundary.replace(microsecond=0)
    return seconds_to_next, next_boundary

# Trading bot logic
def trading_bot():
    global bot_active, position, buy_price, total_profit, pause_duration, pause_start, latest_signal, conn, last_valid_price, stop_time
    if conn is None:
        logger.error("Database connection not initialized. Cannot start trading bot.")
        return

    if not delete_webhook():
        logger.error("Cannot proceed with polling due to persistent webhook. Exiting trading bot.")
        return

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
            last_valid_price = df.iloc[-1][['Open', 'High', 'Low', 'Close', 'Volume']]
            last_valid_price['diff'] = last_valid_price['Close'] - last_valid_price['Open']
            break
        except Exception as e:
            logger.error(f"Error fetching historical data (attempt {attempt + 1}/3): {e}")
            if attempt < 2:
                time.sleep(5)
            else:
                logger.error(f"Failed to fetch historical data for {SYMBOL}.")
                return

    interval_seconds = os.getenv{inter_seconds} # 60  Force 1-minute updates
    logger.info(f"Using interval of {interval_seconds} seconds for timeframe {TIMEFRAME}")

    seconds_to_next, next_boundary = align_to_next_boundary(interval_seconds)
    if seconds_to_next > 0:
        logger.info(f"Waiting {seconds_to_next:.2f} seconds to align with next boundary at {next_boundary.strftime('%Y-%m-%d %H:%M:%S')}")
        time.sleep(seconds_to_next)
    logger.info(f"Bot aligned to boundary at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    stats_interval = 300
    last_stats_time = datetime.now()

    while True:
        if datetime.now() >= stop_time:
            with bot_lock:
                bot_active = False
                if position == "long":
                    latest_data = get_simulated_price()
                    if not pd.isna(latest_data['Close']):
                        profit = latest_data['Close'] - buy_price
                        total_profit += profit
                        signal = create_signal("sell", latest_data['Close'], latest_data, df, profit, total_profit, "Bot stopped due to time limit")
                        store_signal(signal)
                        send_telegram_message(signal, BOT_TOKEN, CHAT_ID)
                        logger.info(f"Generated signal on stop: {signal['action']} at {signal['price']}")
                        latest_signal = signal
                    position = None
                logger.info("Bot stopped due to time limit")
            break

        if not bot_active:
            time.sleep(10)
            continue

        try:
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

            now = datetime.now()
            seconds_since_epoch = now.timestamp()
            if seconds_since_epoch % interval_seconds > 1:
                seconds_to_next, next_boundary = align_to_next_boundary(interval_seconds)
                logger.debug(f"Cycle misaligned, waiting {seconds_to_next:.2f} seconds for {next_boundary.strftime('%Y-%m-%d %H:%M:%S')}")
                time.sleep(seconds_to_next)

            latest_data = get_simulated_price()
            if pd.isna(latest_data['Close']):
                logger.warning("Skipping cycle due to missing price data.")
                time.sleep(interval_seconds)
                continue
            current_price = latest_data['Close']
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            try:
                updates = bot.get_updates(offset=last_update_id, timeout=10)
                for update in updates:
                    if update.message and update.message.text:
                        text = update.message.text.strip()
                        command_chat_id = update.message.chat.id
                        if text == '/help':
                            bot.send_message(chat_id=command_chat_id, text="Commands: /help, /stop, /stopN, /start, /status, /performance, /count, /stats, /daily")
                        elif text == '/stop':
                            with bot_lock:
                                if bot_active and position == "long":
                                    profit = current_price - buy_price
                                    total_profit += profit
                                    signal = create_signal("sell", current_price, latest_data, df, profit, total_profit, "Bot stopped via Telegram")
                                    store_signal(signal)
                                    send_telegram_message(signal, BOT_TOKEN, CHAT_ID)
                                    logger.info(f"Generated signal on /stop: {signal['action']} at {signal['price']}")
                                    latest_signal = signal
                                    position = None
                                bot_active = False
                            bot.send_message(chat_id=command_chat_id, text="Bot stopped.")
                        elif text.startswith('/stop') and text[5:].isdigit():
                            multiplier = int(text[5:])
                            timeframe_seconds = interval_seconds
                            with bot_lock:
                                pause_duration = multiplier * timeframe_seconds
                                pause_start = datetime.now()
                                if position == "long":
                                    profit = current_price - buy_price
                                    total_profit += profit
                                    signal = create_signal("sell", current_price, latest_data, df, profit, total_profit, "Bot paused via Telegram")
                                    store_signal(signal)
                                    send_telegram_message(signal, BOT_TOKEN, CHAT_ID)
                                    logger.info(f"Generated signal on /stopN: {signal['action']} at {signal['price']}")
                                    latest_signal = signal
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
                        elif text == '/daily':
                            with bot_lock:
                                bot_active = True
                                position = None
                                buy_price = None
                                pause_start = None
                                pause_duration = 0
                                start_time = datetime.now()
                                stop_time = start_time + pd.Timedelta(seconds=STOP_AFTER_SECONDS)
                                bot.send_message(chat_id=command_chat_id, text=f"Bot restarted with stop time: {stop_time.strftime('%Y-%m-%d %H:%M:%S')}")
                        elif text == '/status':
                            status = "active" if bot_active else f"paused for {int(pause_duration - (datetime.now() - pause_start).total_seconds())} seconds" if pause_start else "stopped"
                            bot.send_message(chat_id=command_chat_id, text=status)
                        elif text == '/performance':
                            bot.send_message(chat_id=command_chat_id, text=get_performance())
                        elif text == '/count':
                            bot.send_message(chat_id=command_chat_id, text=get_trade_counts())
                        elif text == '/stats':
                            bot.send_message(chat_id=command_chat_id, text="Trade statistics printed to console. Run display_trade_statistics in a separate cell.")
                    last_update_id = update.update_id + 1
            except telegram.error.TelegramError as e:
                logger.error(f"Error processing Telegram updates: {e}")
                if "Conflict" in str(e):
                    logger.warning("Webhook conflict detected. Attempting to delete webhook again.")
                    if delete_webhook():
                        logger.info("Webhook deleted successfully on retry.")
                    else:
                        logger.error("Failed to resolve webhook conflict. Skipping update cycle.")
                        time.sleep(interval_seconds)
                        continue
            except Exception as e:
                logger.error(f"Unexpected error processing Telegram updates: {e}")

            new_row = pd.DataFrame({
                'Open': [latest_data['Open']],
                'Close': [latest_data['Close']],
                'High': [latest_data['High']],
                'Low': [latest_data['Low']],
                'Volume': [latest_data['Volume']],
                'diff': [latest_data['diff']]
            }, index=[pd.Timestamp.now(tz='UTC')])
            df = pd.concat([df, new_row]).tail(100)
            df = add_technical_indicators(df)

            prev_close = df['Close'].iloc[-1] if len(df) >= 2 else df['Close'].iloc[-1]
            percent_change = ((current_price - prev_close) / prev_close * 100) if prev_close != 0 else 0.0
            recommended_action, stop_loss, take_profit = ai_decision(df, position=position, buy_price=buy_price)

            action = "hold"
            profit = 0
            msg = f"HOLD {SYMBOL} at {current_price:.2f}"
            with bot_lock:
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
            latest_signal = signal
            logger.debug(f"Updated latest_signal: {signal}")
            if bot_active and action != "hold":
                threading.Thread(target=send_telegram_message, args=(signal, BOT_TOKEN, CHAT_ID), daemon=True).start()
                logger.info(f"Generated signal: {signal['action']} at {signal['price']}")

            if (datetime.now() - last_stats_time).total_seconds() >= stats_interval:
                logger.info("Periodic trade statistics would be displayed here. Run display_trade_statistics in a separate cell.")
                last_stats_time = datetime.now()

            seconds_to_next, next_boundary = align_to_next_boundary(interval_seconds)
            if seconds_to_next < 1:
                seconds_to_next += interval_seconds
                next_boundary += timedelta(seconds=interval_seconds)
            logger.debug(f"Sleeping for {seconds_to_next:.2f} seconds until next boundary at {next_boundary.strftime('%Y-%m-%d %H:%M:%S')}")
            time.sleep(seconds_to_next)
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
            seconds_to_next, _ = align_to_next_boundary(interval_seconds)
            time.sleep(seconds_to_next if seconds_to_next > 1 else interval_seconds)

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
        'total_profit': total_profit,
        'ema1': latest['ema1'] if not pd.isna(latest['ema1']) else 0.0,
        'ema2': latest['ema2'] if not pd.isna(latest['ema2']) else 0.0,
        'rsi': latest['rsi'] if not pd.isna(latest['rsi']) else 0.0,
        'k': latest['k'] if not pd.isna(latest['k']) else 0.0,
        'd': latest['d'] if not pd.isna(latest['d']) else 0.0,
        'j': latest['j'] if not pd.isna(latest['j']) else 0.0,
        'diff': latest['diff'] if not pd.isna(latest['diff']) else 0.0,
        'message': msg,
        'timeframe': TIMEFRAME
    }

def store_signal(signal):
    try:
        if conn is None:
            logger.error("Cannot store signal: Database connection is None")
            return
        c = conn.cursor()
        c.execute('''
            INSERT INTO trades (
                time, action, symbol, price, open_price, close_price, volume,
                percent_change, stop_loss, take_profit, profit, total_profit,
                ema1, ema2, rsi, k, d, j, diff, message, timeframe
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal['time'], signal['action'], signal['symbol'], signal['price'],
            signal['open_price'], signal['close_price'], signal['volume'],
            signal['percent_change'], signal['stop_loss'], signal['take_profit'],
            signal['profit'], signal['total_profit'],
            signal['ema1'], signal['ema2'], signal['rsi'],
            signal['k'], signal['d'], signal['j'], signal['diff'],
            signal['message'], signal['timeframe']
        ))
        conn.commit()
        logger.debug("Signal stored successfully")
        upload_to_google_drive('renda_bot.db', 'renda_bot.db', GOOGLE_DRIVE_FOLDER_ID)
    except Exception as e:
        logger.error(f"Error storing signal: {e}")

def get_performance():
    try:
        if conn is None:
            logger.error("Cannot fetch performance: Database connection is None")
            return "Database not initialized."
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM trades")
        trade_count = c.fetchone()[0]
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
        return message
    except Exception as e:
        logger.error(f"Error fetching performance: {e}")
        return f"Error fetching performance data: {str(e)}"

def get_trade_counts():
    try:
        if conn is None:
            logger.error("Cannot fetch trade counts: Database connection is None")
            return "Database not initialized."
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
        if conn is None:
            logger.error("Cannot render index.html: Database connection is None")
            return jsonify({"error": "Database not initialized. Please check server logs."}), 500
        # Check if latest_signal is stale (older than 65 seconds)
        signal_time = datetime.strptime(latest_signal['time'], "%Y-%m-%d %H:%M:%S")
        if (datetime.now() - signal_time).total_seconds() > 65:
            logger.debug("latest_signal is stale, fetching from database")
            db_signal = get_latest_signal_from_db()
            if db_signal:
                latest_signal.update(db_signal)
        logger.debug(f"Rendering index.html with latest_signal: {latest_signal}")
        c = conn.cursor()
        c.execute("SELECT * FROM trades ORDER BY time DESC LIMIT 16")
        trades = [dict(zip([col[0] for col in c.description], row)) for row in c.fetchall()]
        stop_time_str = stop_time.strftime("%Y-%m-%d %H:%M:%S")
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Rendering index.html: status={status}, timeframe={TIMEFRAME}, trades={len(trades)}, signal={latest_signal}")
        return render_template('index.html', signal=latest_signal, status=status, timeframe=TIMEFRAME,
                             trades=trades, stop_time=stop_time_str, current_time=current_time)
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
        if conn is None:
            logger.error("Cannot fetch trades: Database connection is None")
            return jsonify({"error": "Database not initialized."}), 500
        c = conn.cursor()
        c.execute("SELECT * FROM trades ORDER BY time DESC LIMIT 16")
        trades = [dict(zip([col[0] for col in c.description], row)) for row in c.fetchall()]
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

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
