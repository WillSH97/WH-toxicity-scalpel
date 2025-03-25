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
# env vars
from dotenv import load_dotenv
import json
import os
load_dotenv()

HF_TOKEN = os.getenv('HF_TOKEN')
BASE_DIR = os.getenv('BASE_DIR')
MODEL_LIST = json.loads(os.getenv('MODEL_LIST'))
TOKENIZER = os.getenv('TOKENIZER')
DEBERTA_FT_PATH = os.getenv('DEBERTA_FT_PATH')

#login for llama_guard
import huggingface_hub
huggingface_hub.login(token=HF_TOKEN)

from exp_datasets.minipile.load_minipile import sample_minipile_text
from deberta_classifier.deberta_inference import load_deberta_finetune_model, deberta_classify
from detoxify_funcs.detoxify_funcs import detoxify_classify
from farrell.farrell_inference import farrell_lexicon
from llama_guard_inf.llama_guard_moderator import moderate as llamaguard_moderate
from mauve_inf.mauve_engine import mauve_scores
from perplexity.perplexity_engine import  ppl_batched
from pythia.pythia_inference import load_model, pythia_generate_batched
from zeroshot_nli.zeroshot_nli_engine import misogyny_zsnli
import pandas as pd

results = {}
#load deberta classifier
deberta_model, deberta_tokenizer, deberta_device = load_deberta_finetune_model(DEBERTA_FT_PATH) ### CHANGE STRING HERE

#load all necessary data

realToxicityPrompts = pd.read_json(path_or_buf='exp_datasets/RealToxicityPrompts/prompts.jsonl', lines=True)
semEval = pd.read_csv('exp_datasets/semeval/semeval_2023_processed.csv')
eacl_guest_dataset = pd.read_csv('exp_datasets/eacl_guest/eacl_guest_preprocessed.csv')
# data prep
semEval_nonMisog = [semEval['datapoint'][i] for i in range(len(semEval)) if semEval['misogynistic_label'][i]=='not sexist']
semEval_nonMisog_txt = " ".join(semEval_nonMisog)
semEval_Misog = [semEval['datapoint'][i] for i in range(len(semEval)) if semEval['misogynistic_label'][i]=='sexist']
semEval_Misog_txt = " ".join(semEval_Misog)

eacl_nonMisog = [str(eacl_guest_dataset['datapoint'][i]) for i in range(len(eacl_guest_dataset)) if eacl_guest_dataset['misogynistic_label'][i]==0]
eacl_nonMisog_txt = " ".join(eacl_nonMisog)
eacl_Misog = [str(eacl_guest_dataset['datapoint'][i]) for i in range(len(eacl_guest_dataset)) if eacl_guest_dataset['misogynistic_label'][i]==1]
eacl_Misog_txt = " ".join(eacl_Misog)


for model_name in MODEL_LIST:
    temp_model_results = {} #initialise temp results as dictionary
    # load model
    model_dir = os.path.join(BASE_DIR, model_name)
    model, tokenizer, device = load_model(model_dir, TOKENIZER)

    #perplexity
    temp_model_results['perplexity_general'] = ppl_batched(model, tokenizer, sample_minipile_text, batch_size=2, device = 'cuda:0')

    #generation
    # written currently based on the fact that generation is NOT Batched in the default 
    toxic_prompt_outputs= []
    toxic_inputs = [itm for itm in realToxicityPrompts["prompt"] for _ in range(2)] #randomly generate 2 outputs per prompt <------- HOW MANY SHOULD I DO?????
    output = pythia_generate_batched(model, tokenizer, device,  toxic_inputs, temperature=0.1, max_length=128, batch_size=4, device = 'cuda:1')
    toxic_prompt_outputs.extend(output)

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
    deberta_results = deberta_classify(deberta_model, deberta_tokenizer, deberta_device, temp_model_results["toxicity_outputs"]) # inherently batched - can change batch_size param here if reqd.

    #Perplexity
    perplexity_results = {}
    perplexity_results['semEval_nonMisog'] = ppl_batched(model, tokenizer, semEval_nonMisog_txt)
    perplexity_results['semEval_Misog'] = ppl_batched(model, tokenizer, semEval_Misog_txt)
    perplexity_results['eacl_nonMisog'] = ppl_batched(model, tokenizer, eacl_nonMisog_txt)
    perplexity_results['eacl_Misog'] = ppl_batched(model, tokenizer, eacl_Misog_txt)

    temp_model_results["perplexity_misog"] = perplexity_results
    
    #MAUVE
    mauve_results = {}
    mauve_scores(predictions, references)
    mauve_results['semEval_nonMisog'] = mauve_scores(temp_model_results["toxicity_outputs"], semEval_nonMisog)
    mauve_results['semEval_Misog'] = mauve_scores(temp_model_results["toxicity_outputs"], semEval_Misog)
    mauve_results['eacl_nonMisog'] = mauve_scores(temp_model_results["toxicity_outputs"], eacl_nonMisog)
    mauve_results['eacl_Misog'] = mauve_scores(temp_model_results["toxicity_outputs"], eacl_Misog)

    temp_model_results["mauve_misog"] = mauve_results    
        

    # FINAL RESULT WRITE
    clean_model_name = model_name.split("/")[0] #assuming all root dirs here are the correct main name
    results[model_name] = temp_model_results
    #dumping interim outputs in case it takes ages
    with open(f"{clean_model_name}_results.json", 'w') as f:
        json.dump(temp_model_results)
    
#save results
with open('pythia_test_results_total.json', 'w') as f:
    json.dump(results, f)
