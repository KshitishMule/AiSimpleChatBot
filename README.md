## AiSimpleChatBot
🤖 AiSimpleChatBot
AiSimpleChatBot is a basic AI chatbot built using Python and Machine Learning techniques. It uses Natural Language Processing (NLP) to understand user input and generate contextually appropriate responses.

📌 Features
Simple GUI using Tkinter

Intent recognition using a trained neural network

JSON-based intent structure for easy customization

Preprocessing with NLTK (lemmatization, tokenization)

Lightweight and beginner-friendly design

🧠 Tech Stack
Python

Tkinter – for GUI

NLTK – for NLP tasks

TensorFlow / Keras – for building and training the model

NumPy & JSON – for data handling

📸 Screenshot

🚀 How to Run
Clone the repository

bash
Copy
Edit
git clone https://github.com/KshitishMule/AiSimpleChatBot.git
cd AiSimpleChatBot
Install dependencies
Ensure you have Python installed, then run:

bash
Copy
Edit
pip install -r requirements.txt
Train the model (if chatbot_model.h5 does not exist)

bash
Copy
Edit
python train_chatbot.py
Run the chatbot

bash
Copy
Edit
python chatbot_gui.py
📄 Intents File Structure
json
Copy
Edit
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "Hello", "Hey"],
      "responses": ["Hello!", "Hi there!", "Greetings!"]
    },
    ...
  ]
}
You can modify the intents.json file to include your own conversation logic.

🛠 Requirements
Python 3.6+

NLTK

TensorFlow

NumPy

Use the following to install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
🙌 Contributing
Contributions are welcome! If you have suggestions for improvements, feel free to fork the repo and open a pull request.

Your Regards
Kshitish Mule, Dheeraj Malviya
