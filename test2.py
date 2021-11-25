from transformers import ElectraTokenizerFast
toker = ElectraTokenizerFast.from_pretrained('google/electra-small-discriminator')
print(toker('sdfsdf', return_offsets_mapping = True))