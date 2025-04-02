'''
again pretty arbitrary, but it'll help us keep code clean? idk
'''

from transformers import pipeline
def load_ZSNLI_classifier(device: str):
    classifier = pipeline("zero-shot-classification",
                          model="cross-encoder/nli-deberta-v3-base",
                         device=device)
    return classifier

def misogyny_zsnli(classifier, input_str):
    '''
    classifies stirngs as misogynistic or otherwise
    '''
    results = classifier(input_str,
              ['misogynistic'],
              multi_label = True #makes the calc technically more valid for binary class, although it doesn't matter for single class - just being pedantic
              )
    return results