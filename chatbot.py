import random
import json
import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load intents
with open("intents.json", "r") as json_data:
    intents = json.load(json_data)

# Load trained model
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]

all_words = data["all_words"]
tags = data["tags"]

model_state = data["model_state"]

# Initialize model
model = NeuralNet(
    input_size,
    hidden_size,
    output_size
).to(device)

model.load_state_dict(model_state)
model.eval()

bot_name = "Orbit"

print("=" * 50)
print("🤖 Orbit Chatbot Started")
print("Type 'quit' to exit")
print("=" * 50)

while True:

    sentence = input("\nYou: ")

    if sentence.lower() in ["quit", "exit", "bye"]:
        print(f"{bot_name}: Goodbye! Have a nice day.")
        break

    # Tokenize sentence
    sentence_tokens = tokenize(sentence)

    # Convert to bag of words
    X = bag_of_words(sentence_tokens, all_words)

    X = X.reshape(1, X.shape[0])

    X = torch.from_numpy(X).to(device)

    # Get prediction
    output = model(X)

    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)

    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:

        for intent in intents["intents"]:

            if tag == intent["tag"]:

                response = random.choice(intent["response"])

                print(f"{bot_name}: {response}")

    else:

        print(
            f"{bot_name}: Sorry, I couldn't understand that. Please try again."
        )
        