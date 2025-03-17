'''
I need to:

- perplexity over minipile

- GENERATE ON RTP
- save outputs (ordered)

then on the outputs:
- llama-guard
- detoxify

- farrell lexicon
- ZSNLI
- Classifier
- Perplexity on SemEval
- MAUVE on SemEval
- Perplexity on Guest
- MAUVE on Guest
'''

from datasets
from deberta_classifier.deberta_inference import deberta_classify
from detoxify.detoxify_funcs import detoxify_classify
from farrel.farrell_inference import farrell_lexicon
from llama_guard.llama_guard_moderator import moderate as llamaguard_moderate
from mauve.mauve_engine import mauve_scores
from perplexity.perplexity_engine import  ppl
from pythia.pythia_inference import load_model, pythia_generate
from zeroshot_nli.zeroshot_nli_engine import misogyny_zsnli
import os

BASE_DIR = ''
list_of_models = ['']
results = {}

for model_name in list_of_models:
    # load model
    model_dir = os.path.join(BASE_DIR, model_name)
    model, tokenizer, device = load_model(model_dir)
    