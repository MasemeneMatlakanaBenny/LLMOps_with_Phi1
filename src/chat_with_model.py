from configurations import set_device,load_model,load_tokenizer

device=set_device() # set the device to cuda if available else cpu

model=load_model()
tokenizer=load_tokenizer()



def response_model(prompt,model=model,tokenizer=tokenizer,device=device):
    # get the tokenizer and model ids:
    tokenizer.pad_token=tokenizer.eos_token
    model.config.pad_token_id=model.config.eos_token_id


    inputs=tokenizer(prompt,return_tensors="pt").to(device)

    # get the input ids:
    input_ids=inputs.input_ids.to(device)

    # get the attention mask:
    attention_mask=inputs.attention_mask.to(device)

    generated_text=model.generate(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    num_return_sequences=1,
                                    max_num_tokens=100,
                                    use_cache=True,
                                    do_sample=False)
    
    return tokenizer.decode(generated_text[0],skip_special_tokens=True)



