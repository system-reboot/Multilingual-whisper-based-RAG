from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import pipeline

QAtokenizer = AutoTokenizer.from_pretrained("SRDdev/QABERT-small")
QAmodel = AutoModelForQuestionAnswering.from_pretrained("SRDdev/QABERT-small")

text = '''Extractive Question Answering is the task of extracting an answer from a text given a question. An example of a
question-answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune
a model on a SQuAD task, you may leverage the examples/pytorch/question-answering/run_squad.py script.'''
question = "What is extractive question answering?"

def question_answering(text,question):
    ask = pipeline("question-answering", model= QAmodel , tokenizer = QAtokenizer)
    result = ask(question=question, context=text)
    print(f"Answer: '{result['answer']}'")
    return result['answer']

