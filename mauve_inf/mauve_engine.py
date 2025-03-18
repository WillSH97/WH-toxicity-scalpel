'''
literally just copied and pasted from here: https://huggingface.co/spaces/evaluate-metric/mauve

this is also so fucking redundant but I'm just keeping formats consistent bossman.
'''

from evaluate import load
mauve = load('mauve')

def mauve_scores(predictions, references):
    return mauve.compute(predictions=predictions, references=references)