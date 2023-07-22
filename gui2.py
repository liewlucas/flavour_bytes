import tkinter as tk
from tkinter import ttk

# Import your recipe generation function and other required libraries here
from transformers import FlaxAutoModelForSeq2SeqLM
from transformers import AutoTokenizer

# ... (your recipe generation function and other imports)
from transformers import FlaxAutoModelForSeq2SeqLM
from transformers import AutoTokenizer

MODEL_NAME_OR_PATH = "flax-community/t5-recipe-generation"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, use_fast=True)
model = FlaxAutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME_OR_PATH)

prefix = "items: "
# generation_kwargs = {
#     "max_length": 512,
#     "min_length": 64,
#     "no_repeat_ngram_size": 3,
#     "early_stopping": True,
#     "num_beams": 5,
#     "length_penalty": 1.5,
# }
generation_kwargs = {
    "max_length": 512,
    "min_length": 64,
    "no_repeat_ngram_size": 3,
    "do_sample": True,
    "top_k": 60,
    "top_p": 0.95
}


special_tokens = tokenizer.all_special_tokens
tokens_map = {
    "<sep>": "--",
    "<section>": "\n"
}
def skip_special_tokens(text, special_tokens):
    for token in special_tokens:
        text = text.replace(token, "")

    return text

def target_postprocessing(texts, special_tokens):
    if not isinstance(texts, list):
        texts = [texts]

    new_texts = []
    for text in texts:
        text = skip_special_tokens(text, special_tokens)

        for k, v in tokens_map.items():
            text = text.replace(k, v)

        new_texts.append(text)

    return new_texts

def generation_function(texts):
    _inputs = texts if isinstance(texts, list) else [texts]
    inputs = [prefix + inp for inp in _inputs]
    inputs = tokenizer(
        inputs,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="jax"
    )

    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **generation_kwargs
    )
    generated = output_ids.sequences
    generated_recipe = target_postprocessing(
        tokenizer.batch_decode(generated, skip_special_tokens=False),
        special_tokens
    )
    return generated_recipe


# Initialize the items list to an empty list
items = []

def on_ok_click():
    # Get the input text from the Text widget and split it into a list
    input_paragraph = input_text.get("1.0", tk.END)
    items_list = [item.strip() for item in input_paragraph.split(',')]

    # Update the global 'items' list with the input values
    global items
    items = items_list

    # Run the generation_function with the updated 'items' list
    generated = generation_function(items)
    for text in generated:
        sections = text.split("\n")
        for section in sections:
            section = section.strip()

    # Convert the generated outcome to a string and update the output_label
    output_text = "\n".join(generated)
    output_label.config(text=output_text)

def on_exit_click():
    root.quit()

# Create the GUI
root = tk.Tk()
root.title('Recipe Generator')
root.configure(bg='white')

# Create a rounded style for the buttons
s = ttk.Style()
s.configure('Rounded.TButton', borderwidth=0, focuscolor='white', focusthickness=0, font=('Helvetica', 12))

main_frame = ttk.Frame(root, padding=10)
main_frame.pack(expand=True, fill='both')

input_label = ttk.Label(main_frame, text='Enter your Available Ingredients')
input_label.pack(pady=5)

# Use a Text widget for the multi-line input
input_text = tk.Text(main_frame, wrap=tk.WORD, width=30, height=5)
input_text.pack(pady=5)

output_label = ttk.Label(main_frame, text='Our output will go here')
output_label.pack(pady=5)

# Centering the buttons in a new frame
button_frame = ttk.Frame(main_frame)
button_frame.pack(pady=10)

ok_button = ttk.Button(button_frame, text='OK', style='Rounded.TButton', command=on_ok_click)
ok_button.pack(side=tk.LEFT, padx=5)

exit_button = ttk.Button(button_frame, text='Exit', style='Rounded.TButton', command=on_exit_click)
exit_button.pack(side=tk.RIGHT, padx=5)

# Set the background color using the style option
s.configure('Rounded.TButton', background='#87CEEB', foreground='dark blue')

root.mainloop()
