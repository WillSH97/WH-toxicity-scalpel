'''
literally just copied and pasted from here: https://huggingface.co/spaces/evaluate-metric/mauve

this is also so fucking redundant but I'm just keeping formats consistent bossman.
'''

import mauve

def mauve_scores(predictions, references, device):
    return mauve.compute(p_text=predictions, q_text=references, device_id=device)['mauve']