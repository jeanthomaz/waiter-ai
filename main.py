import os
import pandas as pd
import numpy as np
import pickle
import telebot
import openai
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

openai.api_key = os.environ["OPENAI_API_KEY"]
bot = telebot.TeleBot(os.environ["TELEGRAM_KEY"])

pkl_embeddings = 'questions_embeddings.pkl'

csv_questions = pd.read_csv('questions.csv')


# Save and read pkl embeddings
def save_to_pkl(data):
    with open(pkl_embeddings, 'wb') as f:
        return pickle.dump(data, f)


def open_pkl_object():
    with open(pkl_embeddings, 'rb') as f:
        return pickle.load(f)


# Create csv file embeddings and save to pkl
question_embeddings = model.encode(csv_questions['Question'].tolist())
answer_embeddings = model.encode(csv_questions['Answer'].tolist())

data = {}
for i, row in csv_questions.iterrows():
        question = row['Question']
        data[question] = {'question_emb': question_embeddings[i], 'answer_emb': answer_embeddings[i]}

save_to_pkl(data)


def get_best_answer(message):
    data = open_pkl_object()
    questions = list(data.keys())
    question_embeddings = np.array([data[q]['question_emb'] for q in questions])

    user_embedding = model.encode([message])[0]

    # Calculate the cosine similarity between the user embedding and the question embeddings
    similarities = np.dot(question_embeddings, user_embedding) / (
            np.linalg.norm(question_embeddings, axis=1) * np.linalg.norm(user_embedding))

    best_question_index = np.argmax(similarities)

    print(similarities)

    if similarities[best_question_index] < 0.4:
        return [0,
                "Fale que não sabe sobre a questão. Pergunte se pode ajudar em "
                "algo mais?"]

    best_response = csv_questions.iloc[best_question_index]['Answer']

    return [1, best_response]


def create_prompt(message):
    return f"Você é um atendente que está ajudando um cliente do Rei dos Salgadinhos\n\nResponda conforme:{get_best_answer(message)}\n\nQ:{message}\n\nA:"


# Telegram bot
@bot.message_handler(commands=['start'])
def handle_start(message):
    bot.send_message(message.chat.id, "Seja bem vindo! Como posso ajudá-lo? Lembre-se de começar suas mensagens com /p!")


@bot.message_handler(commands=['p'])
def handle_chat(message):
    question = message.text[3:]
    response = openai.Completion.create(model="text-davinci-003",
                                        prompt=create_prompt(question), temperature=0.3,
                                        max_tokens=250,
                                        top_p=1)
    answer = response['choices'][0]['text']
    bot.send_message(message.chat.id, answer)


bot.polling()
