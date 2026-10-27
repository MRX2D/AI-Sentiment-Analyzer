import gradio as gr
import plotly.express as px
from transformers import pipeline

model = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
    top_k=None
)

LABELS = {
    "LABEL_0": "Негативна",
    "LABEL_1": "Нейтральна",
    "LABEL_2": "Позитивна"
}

def analyze_sentiment(text):
    if not text or not text.strip():
        return "Введи текст для аналізу.", None

    results = model(text[:2000])[0]
    names = [LABELS.get(r["label"], r["label"]) for r in results]
    scores = [round(r["score"] * 100, 2) for r in results]

    best = max(results, key=lambda r: r["score"])
    report = (
        f"### Результат аналізу\n\n"
        f"**Тональність:** {LABELS.get(best['label'], best['label'])}\n\n"
        f"**Упевненість:** {best['score'] * 100:.2f}%"
    )

    fig = px.bar(
        x=names, y=scores,
        labels={"x": "Тональність", "y": "Імовірність (%)"},
        title="Аналіз тональності",
        range_y=[0, 100]
    )
    return report, fig

demo = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=5, label="Введи текст"),
    outputs=[gr.Markdown(), gr.Plot()],
    title="AI Sentiment Analyzer",
    description="Аналіз позитивної, нейтральної та негативної тональності тексту"
)

if __name__ == "__main__":
    demo.launch()
