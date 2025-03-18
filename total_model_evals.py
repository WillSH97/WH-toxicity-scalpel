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
from detoxify.detoxify_funcs import detoxify_classify
from farrel.farrell_inference import farrell_lexicon
from llama_guard.llama_guard_moderator import moderate as llamaguard_moderate
from mauve.mauve_engine import mauve_scores
from perplexity.perplexity_engine import  ppl
from pythia.pythia_inference import load_model, pythia_generate
from zeroshot_nli.zeroshot_nli_engine import misogyny_zsnli
import os

BASE_DIR = '/mnt/hpccs01/work/toxicity-scalpel-supercompute/'
list_of_models = ['pythia-12b-deduped/step143000/models--EleutherAI--pythia-12b-deduped/snapshots/6e27c828fd23d786ac9ef143d6bf67740af6298c/']
TOKENIZER = 'pythia-12b-deduped' #in case there's an issue - also assuming all tokenizers are identical for all model sizes.
results = {}

for model_name in list_of_models:
    results[model_name] = {} #initialise results dictionary
    # load model
    model_dir = os.path.join(BASE_DIR, model_name)
    model, tokenizer, device = load_model(model_dir, TOKENIZER)

    #perplexity
    results[model_name]['perplexity_general'] = ppl(model, tokenizer, sample_minipile_text)

    #save results
    with open('pythia_test_results_total.json', 'w') as f:
        json.dump(results, f)