# Fine-tuned-Multilingual-whisper-based-RAG-on-hindi-dataset

A real-time Retrieval-Augmented-Generation(RAG) based model to perform question answering on hindi audio data. Here, the fine-tuned facebook's whisper-tiny model downsampled the word error rate(WER) to 74.24 for the hindi dataset. 


## Deployment Code:

Follow these steps to run the prototype in your system:

1. git clone https://github.com/system-reboot/Multilingual-whisper-based-RAG.git
2. cd Multilingual-whisper-based-RAG
3. jupyter execute fine-tuning-whisper.ipynb
4. python3 run inference.py
   

## Files:

1. fine-tuning-whisper.ipynb - Whisper-tiny model tuned for hindi dataset.
2. rag.py - QA-Bert model for performing question answering on the passed audio.
3. inference.py - Displays the Gradio-based interface for inference results.

   
## Note:
Try to give shorter length audio for efficient results.
