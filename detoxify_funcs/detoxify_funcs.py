'''
This is a little silly, but for the sake of consistency of these tools for now I guess I'll format as such
'''

from detoxify import Detoxify

def detoxify_classify(input_str):
    results = Detoxify('multilingual').predict([input_str])
    return results