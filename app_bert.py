import streamlit as st
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification


# 1. Загрузка модели и токенайзера

@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = BertTokenizerFast.from_pretrained("bert_tokenizer")
    model = BertForSequenceClassification.from_pretrained("bert_model")
    model.eval()
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()
device = torch.device("cpu")
model.to(device)

# 2. Интерфейс Streamlit

st.set_page_config(page_title="🎭 Анализ отзывов (BERT)", page_icon="🤖", layout="centered")

st.title("🤖 Анализ тональности комментариев с помощью BERT")
st.markdown("Введите текст отзыва, и модель определит его **настроение** 😊 или 😡")

text = st.text_area("✍️ Введите текст:", height=150)

if st.button("🔍 Анализировать"):
    if not text.strip():
        st.warning("Введите текст для анализа!")
    else:
        inputs = tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        st.markdown("---")
        if pred == 1:
            st.success("😊 **Положительный отзыв**")
        else:
            st.error("😡 **Отрицательный отзыв**")

st.markdown("---")
st.caption("Проект: Анализ тональности на IMDB — Мукажанов Адиль 🎓")

#streamlit run app_bert.py