import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
import json
import pickle
import os
import tkinter as tk
from tkinter import Canvas, Entry, Frame, Label, Scrollbar
import datetime
import requests

nltk.download('punkt')
stemmer = LancasterStemmer()

with open("intents.json", encoding='utf-8') as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except FileNotFoundError:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])
        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(set(words))
    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = [1 if stemmer.stem(w.lower()) in [stemmer.stem(wd.lower()) for wd in doc] else 0 for w in words]
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

model = keras.Sequential([
    layers.Input(shape=(len(training[0]),)),
    layers.Dense(12, activation='relu'),
    layers.Dense(12, activation='relu'),
    layers.Dense(len(output[0]), activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

if os.path.exists("model.weights.h5"):
    try:
        model.load_weights("model.weights.h5")
    except Exception:
        print("Weight loading failed. Re-training...")
        os.remove("model.weights.h5")
        model.fit(training, output, epochs=1200, batch_size=8, verbose=1)
        model.save_weights("model.weights.h5")
else:
    model.fit(training, output, epochs=1200, batch_size=8, verbose=1)
    model.save_weights("model.weights.h5")

def serpapi_search(query):
    api_key = "Your API KEY"
    url = f"https://serpapi.com/search.json?q={query}&api_key={api_key}&hl=en&gl=us"
    try:
        response = requests.get(url)
        data = response.json()
        if "answer_box" in data:
            ab = data["answer_box"]
            if "answer" in ab:
                return ab["answer"]
            elif "snippet" in ab:
                return ab["snippet"]
            elif "highlighted_words" in ab:
                return ", ".join(ab["highlighted_words"])
        if "organic_results" in data and len(data["organic_results"]) > 0:
            result = data["organic_results"][0]
            snippet = result.get("snippet", "")
            title = result.get("title", "")
            link = result.get("link", "")
            return f"{title}:\n{snippet}\n\n🔗 {link}" if snippet else "No snippet found."
        else:
            return "Hmm... couldn't find anything interesting for that. Try another one!"
    except Exception as e:
        return f"Oops! Error occurred: {str(e)}"

def bag_of_words(s, words):
    bag = [0] * len(words)
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(w.lower()) for w in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)

def get_response(msg):
    results = model.predict(np.array([bag_of_words(msg, words)]), verbose=0)[0]
    results_index = np.argmax(results)
    confidence = results[results_index]

    if confidence > 0.7:
        tag = labels[results_index]
    else:
        return serpapi_search(msg)

    if tag == "time":
        return f"🕒 It's currently {datetime.datetime.now().strftime('%H:%M:%S')}!"

    for tg in data["intents"]:
        if tg["tag"] == tag:
            return random.choice(tg["responses"])

    return "🤔 I'm still learning. Could you try rephrasing that?"

# GUI Setup
window = tk.Tk()
window.title("💬 AI ChatBot Companion")
window.geometry("600x700")
window.configure(bg="#ffffff")
window.resizable(False, False)

header = Frame(window, bg="#0d47a1", height=60)
header.pack(fill=tk.X)
Label(header, text="Hi there! 👋 Ready to chat?", bg="#0d47a1", fg="white", font=("Segoe UI", 14, "bold"), anchor="w", padx=15).pack(fill=tk.BOTH, pady=10)

chat_frame = Frame(window, bg="#ffffff")
chat_frame.pack(fill=tk.BOTH, expand=True)

chat_canvas = Canvas(chat_frame, bg="#ffffff", bd=0, highlightthickness=0)
scrollbar = Scrollbar(chat_frame, orient="vertical", command=chat_canvas.yview)
chat_canvas.configure(yscrollcommand=scrollbar.set)

scroll_frame = Frame(chat_canvas, bg="#ffffff")
chat_canvas.create_window((0, 0), window=scroll_frame, anchor='nw')

def configure_scroll(event):
    chat_canvas.configure(scrollregion=chat_canvas.bbox("all"))

scroll_frame.bind("<Configure>", configure_scroll)

chat_canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

chat_log = scroll_frame

input_frame = Frame(window, bg="#ffffff")
input_frame.pack(fill=tk.X, pady=10, padx=10)

user_entry = Entry(input_frame, font=("Segoe UI", 12), bg="#f1f1f1", bd=0)
user_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 10))

def send(event=None):
    user_input = user_entry.get().strip()
    if not user_input:
        return
    add_message("You", user_input, "user")
    user_entry.delete(0, tk.END)

    placeholder = add_message("Bot", "Typing...", "bot")
    window.after(500, lambda: add_bot_response(user_input, placeholder))

def add_bot_response(user_input, typing_label):
    response = get_response(user_input)
    typing_label.config(text=response)
    chat_canvas.update_idletasks()
    chat_canvas.yview_moveto(1.0)

def add_message(sender, msg, msg_type):
    frame = Frame(chat_log, bg="#ffffff", pady=5)
    timestamp = datetime.datetime.now().strftime("%H:%M")

    bubble = Label(frame, text=f"{msg}\n🕒 {timestamp}",
                   bg="#DCF8C6" if msg_type == "user" else "#F1F0F0",
                   fg="#000", wraplength=550,
                   justify="right" if msg_type == "user" else "left",
                   font=("Segoe UI", 10), padx=10, pady=6)

    bubble.pack(anchor="e" if msg_type == "user" else "w", padx=10, fill="x")
    frame.pack(fill=tk.X, anchor="w", padx=10)
    chat_canvas.update_idletasks()
    chat_canvas.yview_moveto(1.0)
    return bubble

user_entry.bind("<Return>", send)

send_btn = tk.Button(input_frame, text="➡", font=("Segoe UI", 14), bg="#0d47a1", fg="white", command=send)
send_btn.pack(side=tk.RIGHT, padx=(10, 0), ipadx=10, ipady=4)

add_message("Bot", "🧠 Hello! Ask me anything — and I'll find the answer, even on the web!", "bot")
user_entry.focus()

window.mainloop()
