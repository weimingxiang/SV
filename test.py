from m_albert import AlbertModel
import torch
model = AlbertModel.from_pretrained("albert-base-v2")


a = torch.rand(10, 13)
b = torch.rand(20, 13)
c = torch.rand(15, 13)
input_ids = [a, b, c]

model(input_ids,
       attention_mask=torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
       token_type_ids=torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
       # encoder_attention_mask=mask,
       output_hidden_states=True,
       return_dict=False)