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
import argparse

load_dotenv()

HF_TOKEN = os.getenv('HF_TOKEN')
BASE_DIR = os.getenv('BASE_DIR')
# MODEL_LIST = json.loads(os.getenv('MODEL_LIST'))
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
from llama_guard_inf.llama_guard_moderator import load_llama_guard_model
from mauve_inf.mauve_engine import mauve_scores
from perplexity.perplexity_engine import  ppl_batched
from pythia.pythia_inference import load_model, pythia_generate_batched
from zeroshot_nli.zeroshot_nli_engine import load_ZSNLI_classifier, misogyny_zsnli
import pandas as pd
import concurrent.futures
from copy import deepcopy
import gc
import torch

results = {}
#load deberta classifier
deberta_model, deberta_tokenizer, deberta_device = load_deberta_finetune_model(DEBERTA_FT_PATH, device = 'cuda:3') ### CHANGE STRING HERE - GPU_4 for now

# load llama guard model
llamaguard_model, llamaguard_tokenizer, llamaguard_device = load_llama_guard_model('cuda:2') #placing it on GPU_3 for now

# load ZSNLI classifier
ZSNLI_classifier = load_ZSNLI_classifier(device = 'cuda:3') #placing it on GPU_4 for now

#load all necessary data

realToxicityPrompts = pd.read_json(path_or_buf='exp_datasets/RealToxicityPrompts/prompts.jsonl', lines=True)
semEval = pd.read_csv('exp_datasets/semeval/semeval_2023_processed.csv')
eacl_guest_dataset = pd.read_csv('exp_datasets/eacl_guest/eacl_guest_preprocessed.csv')
# data prep
semEval_nonMisog = [str(semEval['datapoint'][i]) for i in range(len(semEval)) if semEval['misogynistic_label'][i]=='not sexist']
semEval_nonMisog_txt = " ".join(semEval_nonMisog)
semEval_Misog = [str(semEval['datapoint'][i]) for i in range(len(semEval)) if semEval['misogynistic_label'][i]=='sexist']
semEval_Misog_txt = " ".join(semEval_Misog)

eacl_nonMisog = [str(eacl_guest_dataset['datapoint'][i]) for i in range(len(eacl_guest_dataset)) if eacl_guest_dataset['misogynistic_label'][i]==0]
eacl_nonMisog_txt = " ".join(eacl_nonMisog)
eacl_Misog = [str(eacl_guest_dataset['datapoint'][i]) for i in range(len(eacl_guest_dataset)) if eacl_guest_dataset['misogynistic_label'][i]==1]
eacl_Misog_txt = " ".join(eacl_Misog)

#define multiprocess batch 1
def general_ppl_and_textgen(model, tokenizer, sample_minipile_text, realToxicityPrompts):
    # Use concurrent.futures to run perplexity and generation concurrently
    with concurrent.futures.ThreadPoolExecutor() as executor:

        #making deepcopies of models so that they're separate
        ppl_model = deepcopy(model)
        ppl_model.to('cuda:0')
        ppl_model.eval()
        ppl_tokenizer=deepcopy(tokenizer)

        gen_model1 = deepcopy(model)
        gen_model1.to('cuda:1')
        gen_model1.eval()
        gen_tokenizer1=deepcopy(tokenizer)

        gen_model2 = deepcopy(model)
        gen_model2.to('cuda:2')
        gen_model2.eval()
        gen_tokenizer2=deepcopy(tokenizer)

        gen_model3 = deepcopy(model)
        gen_model3.to('cuda:3')
        gen_model3.eval()
        gen_tokenizer3=deepcopy(tokenizer)
        
        
        
        # Submit perplexity calculation as a future
        ppl_future = executor.submit(ppl_batched, ppl_model, ppl_tokenizer, str(sample_minipile_text), 1, 'cuda:0')
        
        # Prepare generation inputs (2 outputs per prompt)
        toxic_inputs = [str(itm) for itm in realToxicityPrompts["prompt"]]
        
        # Submit generation tasks - it's ugly but I don't want to try something smart in prod and then realise it doesn't work after 24h
        generation_future1 = executor.submit(
            pythia_generate_batched, 
            gen_model1, 
            gen_tokenizer1, 
            'cuda:1',
            list(toxic_inputs),
            8,
            0.1, 
            128,  
        )
        generation_future2 = executor.submit(
            pythia_generate_batched, 
            gen_model2, 
            gen_tokenizer2, 
            'cuda:2',
            list(toxic_inputs),
            8,
            0.1, 
            128,  
        )
        generation_future3 = executor.submit(
            pythia_generate_batched, 
            gen_model3, 
            gen_tokenizer3, 
            'cuda:3',
            list(toxic_inputs),
            8,
            0.1, 
            128,  
        )
        
        # Wait for and collect results
        temp_model_results = {}
        
        # Get perplexity result
        temp_model_results['perplexity_general'] = ppl_future.result()
        
        # Get generation outputs - it's ugly but I don't want to try something smart in prod and then realise it doesn't work after 24h
        total_generation_results = []
        generation_results1 = generation_future1.result()
        generation_results2 = generation_future2.result()
        generation_results3 = generation_future3.result()
        total_generation_results.extend(generation_results1)
        total_generation_results.extend(generation_results2)
        total_generation_results.extend(generation_results3)

        temp_model_results["toxicity_outputs"] = total_generation_results

        #clean models
        ppl_model.to('cpu')
        del ppl_model
        gen_model1.to('cpu')
        del gen_model1
        gen_model2.to('cpu')
        del gen_model2
        gen_model3.to('cpu')
        del gen_model3
        gc.collect()
        torch.cuda.empty_cache()

        #FOR DEBUG
        print(torch.cuda.memory_summary())
        
        return temp_model_results

def parallel_output_analysis(model, tokenizer, temp_model_results):
    # Use concurrent.futures to run perplexity and generation concurrently
    with concurrent.futures.ThreadPoolExecutor() as executor:
        #making deepcopies of models so that they're separate
        ppl_model = deepcopy(model)
        ppl_model.to('cuda:0')
        ppl_model.eval()
        ppl_tokenizer = deepcopy(tokenizer)

        #llama_guard concurrency
        llama_guard_futures = []
        for output in temp_model_results["toxicity_outputs"]:
            llama_guard_futures.append(executor.submit(llamaguard_moderate, output, llamaguard_model, llamaguard_tokenizer, llamaguard_device))

        #detoxify concurrency
        detoxify_futures = []
        for output in temp_model_results["toxicity_outputs"]:
            detoxify_futures.append(executor.submit(detoxify_classify, output))
    
        #farrell lexicon concurrency
        farrell_futures = []
        for output in temp_model_results["toxicity_outputs"]:
            farrell_futures.append(executor.submit(farrell_lexicon, output))
    
        #ZSNLI
        ZSNLI_futures = []
        for output in temp_model_results["toxicity_outputs"]:
            ZSNLI_futures.append(executor.submit(misogyny_zsnli, ZSNLI_classifier, output))

        #deberta classifier
        deberta_future = executor.submit(deberta_classify, deberta_model, deberta_tokenizer, deberta_device, temp_model_results["toxicity_outputs"]) # inherently batched - can change batch_size param here if reqd.

        #MAUVE
        mauve_semeval_nonmisog_futures = executor.submit(mauve_scores, list(temp_model_results["toxicity_outputs"]), semEval_nonMisog, 1)
        mauve_semeval_misog_futures = executor.submit(mauve_scores, list(temp_model_results["toxicity_outputs"]), semEval_Misog, 1)
        mauve_eacl_nonmisog_futures = executor.submit(mauve_scores, list(temp_model_results["toxicity_outputs"]), eacl_nonMisog, 1)
        mauve_eacl_misog_futures = executor.submit(mauve_scores, list(temp_model_results["toxicity_outputs"]), eacl_Misog, 1)

        
        #Perplexity
        
        ppl_semeval_nonmisog_futures = executor.submit(ppl_batched, ppl_model, ppl_tokenizer, semEval_nonMisog_txt, 1, 'cuda:0')
        ppl_semeval_nonmisog_results = ppl_semeval_nonmisog_futures.result() # clearing GPU vram here?
        ppl_semeval_misog_futures = executor.submit(ppl_batched, ppl_model, ppl_tokenizer, semEval_Misog_txt, 1, 'cuda:0')
        ppl_semeval_misog_results = ppl_semeval_misog_futures.result() # clearing GPU vram here?
        ppl_eacl_nonmisog_futures = executor.submit(ppl_batched, ppl_model, ppl_tokenizer, eacl_nonMisog_txt, 1, 'cuda:0')
        ppl_eacl_nonmisog_results = ppl_eacl_nonmisog_futures.result() # clearing GPU vram here?
        ppl_eacl_misog_futures = executor.submit(ppl_batched, ppl_model, ppl_tokenizer, eacl_Misog_txt, 1, 'cuda:0')
        ppl_eacl_misog_results = ppl_eacl_misog_futures.result() # clearing GPU vram here?
        


        # gather all results:
        llama_guard_results = [future.result() for future in llama_guard_futures] #<---- TEMPORARY PLACEMENT DURING DEV - MOVE TO THE END OF FUNC FOR DEPLOYMENT
        temp_model_results["llama_guard"] = llama_guard_results #<---- TEMPORARY PLACEMENT DURING DEV - MOVE TO THE END OF FUNC FOR DEPLOYMENT

        detoxify_results = [future.result() for future in detoxify_futures]
        temp_model_results["detoxify"] = detoxify_results

        farrell_results = [future.result() for future in farrell_futures]
        temp_model_results["farrell"] = farrell_results

        ZSNLI_results = [future.result() for future in ZSNLI_futures]
        temp_model_results["ZSNLI"] = ZSNLI_results

        deberta_results = deberta_future.result()
        temp_model_results["deberta_classifier"] = deberta_results

        perplexity_results = {
            'semEval_nonMisog': ppl_semeval_nonmisog_results,
            'semEval_Misog': ppl_semeval_misog_results,
            'eacl_nonMisog': ppl_eacl_nonmisog_results,
            'eacl_Misog': ppl_eacl_misog_results,
        }
    
        temp_model_results["perplexity_misog"] = perplexity_results

        mauve_results = {
            'semEval_nonMisog': mauve_semeval_nonmisog_futures.result(),
            'semEval_Misog': mauve_semeval_misog_futures.result(),
            'eacl_nonMisog': mauve_eacl_nonmisog_futures.result(),
            'eacl_Misog': mauve_eacl_misog_futures.result(),
        }
    
        temp_model_results["mauve_misog"] = mauve_results
        
        #clean models
        ppl_model.to('cpu')
        del ppl_model
        gc.collect()
        torch.cuda.empty_cache()

        return temp_model_results

def main(model_name):
    temp_model_results = {} #initialise temp results as dictionary
    # load model
    model_dir = os.path.join(BASE_DIR, model_name)
    model, tokenizer, temp_device = load_model(model_dir, TOKENIZER)
    tokenizer.pad_token = tokenizer.eos_token # FOR BUG

    model.to('cpu') #seems to be causing issues with memory management

    # Run concurrent tasks
    temp_model_results = general_ppl_and_textgen(model, tokenizer, sample_minipile_text, realToxicityPrompts)

    # temp_model_results["mauve_misog"] = mauve_results    
    temp_model_results = parallel_output_analysis(model, tokenizer, temp_model_results)    

    # FINAL RESULT WRITE
    clean_model_name = model_name.split("/")[0] #assuming all root dirs here are the correct main name
    results[model_name] = temp_model_results
    #dumping interim outputs in case it takes ages
    with open(f"{clean_model_name}_results.json", 'w') as f:
        json.dump(temp_model_results, f)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--modeldir", help="model dir to run model from", type=str)
    args = parser.parse_args()
    main(args.modeldir)
# #save results
# with open('pythia_test_results_total.json', 'w') as f:
#     json.dump(results, f)
