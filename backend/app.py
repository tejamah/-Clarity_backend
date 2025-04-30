import eventlet
eventlet.monkey_patch()

from flask import Flask, jsonify, request
from flask_socketio import SocketIO
from flask_cors import CORS
import threading
import time
import feedparser
import os

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# === Flask Setup ===
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# === Model Setup ===
tokenizer = AutoTokenizer.from_pretrained("teja00007/model-name")
model = AutoModelForSeq2SeqLM.from_pretrained("teja00007/model-name")

# === RSS Feed Categories ===
rss_feeds = {
    "general": "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en",
    "technology": "https://news.google.com/rss/search?q=technology&hl=en-US&gl=US&ceid=US:en",
    "business": "https://news.google.com/rss/search?q=business&hl=en-US&gl=US&ceid=US:en",
    "sports": "https://news.google.com/rss/search?q=sports&hl=en-US&gl=US&ceid=US:en",
    "entertainment": "https://news.google.com/rss/search?q=entertainment&hl=en-US&gl=US&ceid=US:en"
}

# === Global News Cache ===
news_cache = {}

# === Summarization Endpoint ===
@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        output = model.generate(**inputs, max_new_tokens=100)
        summary = tokenizer.decode(output[0], skip_special_tokens=True)
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === Internal Summarization ===
def summarize_internal(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        output = model.generate(**inputs, max_new_tokens=100)
        return {"summary": tokenizer.decode(output[0], skip_special_tokens=True)}
    except:
        return {"summary": "Summary unavailable."}

# === News Fetcher ===
def fetch_and_summarize_news():
    global news_cache
    new_data = {}

    print("[+] Fetching and summarizing news...")

    for category, url in rss_feeds.items():
        try:
            feed = feedparser.parse(url)
            articles = []

            for entry in feed.entries[:20]:
                title = entry.get("title", "Untitled")
                description = entry.get("summary", title)
                summary = summarize_internal(description).get("summary", "")

                articles.append({
                    "title": title,
                    "summary": summary,
                    "url": entry.get("link", ""),
                    "image": ""  # No images from RSS
                })

            new_data[category] = articles

        except Exception as e:
            print(f"[{category}] Error: {e}")

    news_cache = new_data
    socketio.emit("news_update", news_cache)
    print("[+] Sent summarized news to all clients.")

# === Scheduler ===
def schedule_updates():
    while True:
        fetch_and_summarize_news()
        time.sleep(300)

# === Routes ===
@app.route("/")
def home():
    return jsonify({"message": "AI News Summarizer API is live!"})

@app.route("/news/<category>")
def get_news(category):
    return jsonify(news_cache.get(category, []))

@socketio.on("connect")
def handle_socket_connect():
    print("[+] A new client connected.")

# === Start the App ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Dynamic for Render
    threading.Thread(target=schedule_updates, daemon=True).start()
    fetch_and_summarize_news()
    socketio.run(app, host="0.0.0.0", port=port)
