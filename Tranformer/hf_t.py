from transformers import pipeline 
import torch

device = torch.cuda.get_device_name()
translator = pipeline(task = "text2text-generation", model="facebook/m2m100_418M", device=device)
print(translator)

print(translator('This is a translation made easy with transformers', forced_bos_token_id = translator.tokenizer.get_lang_id(lang='hi')))
