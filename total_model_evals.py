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
from mauve_inf.mauve_engine import mauve_scores
from perplexity.perplexity_engine import  ppl
from pythia.pythia_inference import load_model, pythia_generate
from zeroshot_nli.zeroshot_nli_engine import misogyny_zsnli
import os
import pandas as pd
import json

BASE_DIR = ''
list_of_models = ['']
TOKENIZER = '' #in case there's an issue - also assuming all tokenizers are identical for all model sizes.
results = {}

#load all necessary data

realToxicityPrompts = pd.read_csv(path_or_buf='exp_datasets/RealToxicityPrompts/prompts.jsonl', lines=True)
semEval = pd.read_csv('exp_datasets/semeval/semeval_2023_processed.csv')
eacl_guest_dataset = pd.read_csv('exp_datasets/eacl_guest/eacl_guest_preprocessed.csv')
# data prep
semEval_nonMisog = [semEval['datapoint'][i] for i in range(len(semEval)) if semEval['misogynistic_label'][i]=='not sexist']
semEval_nonMisog_txt = " ".join(semEval_nonMisog)
semEval_Misog = [semEval['datapoint'][i] for i in range(len(semEval)) if semEval['misogynistic_label'][i]=='sexist']
semEval_Misog_txt = " ".join(semEval_Misog)

eacl_nonMisog = [eacl_guest_dataset['datapoint'][i] for i in range(len(eacl_guest_dataset)) if eacl_guest_dataset['misogynistic_label'][i]==0]
eacl_nonMisog_txt = " ".join(eacl_nonMisog)
eacl_Misog = [eacl_guest_dataset['datapoint'][i] for i in range(len(eacl_guest_dataset)) if eacl_guest_dataset['misogynistic_label'][i]==1]
eacl_Misog_txt = " ".join(eacl_Misog)

for model_name in list_of_models:
    temp_model_results = {} #initialise temp results as dictionary
    # load model
    model_dir = os.path.join(BASE_DIR, model_name)
    model, tokenizer, device = load_model(model_dir, TOKENIZER)

    #perplexity
    temp_model_results['perplexity_general'] = ppl(model, tokenizer, sample_minipile_text)

    #generation
    # written currently based on the fact that generation is NOT Batched in the default 
    toxic_prompt_outputs= []
    for prompt in realToxicityPrompts["prompt"]:
        for i in range(3): #randomly generate 3 outputs per prompt <------- HOW MANY SHOULD I DO?????
            output = pythia_generate(model, tokenizer, device, prompt['text'], temperature=0.1, max_length=128)
            toxic_prompt_outputs.append(output)

    temp_model_results["toxicity_outputs"] = toxic_prompt_outputs

    #llama_guard
    llama_guard_results = []
    for output in temp_model_results["toxicity_outputs"]:
        result = llamaguard_moderate(output)
        llama_guard_results.append(result)

    temp_model_results["llama_guard"] = llama_guard_results

    #detoxify
    detoxify_results = []
    for output in temp_model_results["toxicity_outputs"]:
        result = detoxify_classify(output)
        detoxify_results.append(result)

    temp_model_results["detoxify"] = detoxify_results

    #farrell lexicon
    farrell_results = []
    for output in temp_model_results["toxicity_outputs"]:
        result = farrell_lexicon(output)
        farrell_results.append(result)

    temp_model_results["farrell"] = farrell_results

    #ZSNLI
    ZSNLI_results = []
    for output in temp_model_results["toxicity_outputs"]:
        result = misogyny_zsnli(output)
        farrell_results.append(result)

    temp_model_results["ZSNLI"] = ZSNLI_results

    #deberta classifier
    deberta_results = deberta_classify(temp_model_results["toxicity_outputs"]) # inherently batched - can change batch_size param here if reqd.

    #Perplexity
    perplexity_results = {}
    perplexity_results['semEval_nonMisog'] = ppl(model, tokenizer, semEval_nonMisog_txt)
    perplexity_results['semEval_Misog'] = ppl(model, tokenizer, semEval_Misog_txt)
    perplexity_results['eacl_nonMisog'] = ppl(model, tokenizer, eacl_nonMisog_txt)
    perplexity_results['eacl_Misog'] = ppl(model, tokenizer, eacl_Misog_txt)

    temp_model_results["perplexity_misog"] = perplexity_results
    
    #MAUVE
    mauve_results = {}
    mauve_scores(predictions, references)
    mauve_results['semEval_nonMisog'] = mauve_scores(temp_model_results["toxicity_outputs"] semEval_nonMisog_txt)
    mauve_results['semEval_Misog'] = mauve_scores(temp_model_results["toxicity_outputs"] semEval_Misog_txt)
    mauve_results['eacl_nonMisog'] = mauve_scores(temp_model_results["toxicity_outputs"] eacl_nonMisog_txt)
    mauve_results['eacl_Misog'] = mauve_scores(temp_model_results["toxicity_outputs"] eacl_Misog_txt)

    temp_model_results["mauve_misog"] = mauve_results    
        

    # FINAL RESULT WRITE
    results[model_name] = temp_model_results







#save results
with open('pythia_test_results_total.json', 'w') as f:
    json.dump(results, f)
