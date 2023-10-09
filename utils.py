from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from accelerate import Accelerator

model = AutoModelForQuestionAnswering.from_pretrained("model_outputs/checkpoint-1000")
tokenizer = AutoTokenizer.from_pretrained("tokenizer")
accelerator = Accelerator()


def model_predict(context, question):
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt")
    # print("inputs", inputs)
    # print("inputs", type(inputs))
    input_ids = inputs["input_ids"].tolist()[0]
    device = "cuda"
    device =  accelerator.device
    inputs.to(device)

    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer_model = model(**inputs)
    

    answer_start = torch.argmax(
        answer_model['start_logits']
    )  # Get the most likely beginning of answer with the argmax of the score
    answer_end = torch.argmax(answer_model['end_logits']) + 1  # Get the most likely end of answer with the argmax of the score

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return answer


#print(model_predict(text, question))