# %%
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# %%
sentence = "Transformers are powerful models the bigger the better. I love machine learning!"
inputs = tokenizer(sentence, return_tensors="pt")
print(inputs)
print(type(inputs))

# %%
print(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))
# %%
from transformers import AutoModel
model = AutoModel.from_pretrained("bert-base-uncased")
print(type(model))
#%%
print(inputs)
# %%
outputs = model(**inputs)
# %%
print(outputs.last_hidden_state.shape)

# %%
