# 🤖 Sentiment Analysis using BERT

This project performs **sentiment analysis** on movie reviews from the **IMDB dataset**,  
classifying each text as **Positive 😊** or **Negative 😠** using a fine-tuned **BERT** model.

## 🧠 Technologies Used
- Python 3.12  
- PyTorch  
- Hugging Face Transformers (BERT)  
- Streamlit  
- IMDB Dataset

## ⚙️ How It Works
1. The BERT model is fine-tuned on labeled IMDB reviews.  
2. The trained model and tokenizer are saved locally (`bert_model/`, `bert_tokenizer/`).  
3. Streamlit provides a web interface where users can enter text and get instant predictions.

## 🚀 Run Locally
To run this app on your computer:

```bash
pip install -r requirements.txt
streamlit run app_bert.py
