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

from exp_datasets.minipile.load_minipile import sample_minipile_text
from deberta_classifier.deberta_inference import deberta_classify
from detoxify_funcs.detoxify_funcs import detoxify_classify
from farrell.farrell_inference import farrell_lexicon
from llama_guard_inf.llama_guard_moderator import moderate as llamaguard_moderate
from mauve.mauve_engine import mauve_scores
from perplexity.perplexity_engine import  ppl
from pythia.pythia_inference import load_model, pythia_generate
from zeroshot_nli.zeroshot_nli_engine import misogyny_zsnli
import os

BASE_DIR = '/Documents/Git/misc-NOTGIT/pythiatest/toxicity-scalpel-supercompute'
list_of_models = ['pythia-70m-deduped/step143000/models--EleutherAI--pythia-70m-deduped/snapshots/4ad6c938b037fd4762343dcc441ba1012a7401c8']
TOKENIZER = '/Documents/Git/misc-NOTGIT/pythiatest/toxicity-scalpel-supercompute/pythia-70m-deduped/step143000/models--EleutherAI--pythia-70m-deduped/snapshots/4ad6c938b037fd4762343dcc441ba1012a7401c8' #in case there's an issue - also assuming all tokenizers are identical for all model sizes.
results = {}

for model_name in list_of_models:
    temp_model_results
    results[model_name] = {} #initialise results dictionary
    # load model
    model_dir = os.path.join(BASE_DIR, model_name)
    model, tokenizer, device = load_model(model_dir, TOKENIZER)

    #perplexity
    results[model_name]['perplexity_general'] = ppl(model, tokenizer, sample_minipile_text)

    results
    







#save results
with open('pythia_test_results_total.json', 'w') as f:
    json.dump(results, f)
