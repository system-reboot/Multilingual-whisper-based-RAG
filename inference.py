from transformers import pipeline
import gradio as gr
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from rag import question_answering

tmodel = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
ttokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

pipe = pipeline(model="san457/whisper-tiny-hi")  

def translate(text):
    # translate Hindi to English
    ttokenizer.src_lang = "hi"
    encoded_hi = ttokenizer(text, return_tensors="pt")
    generated_tokens = tmodel.generate(**encoded_hi, forced_bos_token_id=ttokenizer.get_lang_id("en"))
    return ttokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    
def transcribe(audio):
    print(audio)
    text = pipe(audio)["text"]
    translated_text = translate(text)
    return translated_text

def ques_ans(audio,question):
    translated_text = transcribe(audio)
    answer = question_answering(translated_text[0],question)
    # print(translated_text)
    return answer

iface = gr.Interface(
    fn=ques_ans, 
    inputs=[gr.Audio(type="filepath"),gr.Textbox(label="Enter your question")], 
    outputs="text",
    title="Whisper Tiny Hindi",
    description="Realtime demo for Hindi speech recognition and translation using a fine-tuned Whisper tiny model and facebook's m2m100 model.",
)

iface.launch()