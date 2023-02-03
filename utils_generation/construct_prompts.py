import numpy as np
import pandas as pd
import random
from promptsource.templates import DatasetTemplates
from copy import deepcopy



filter_length = 500

label_dict = {
    "imdb": ["negative", "positive"], # This is for normal IMDB
    "amazon-polarity": ["negative", "positive"],
    "ag-news": ["politics", "sports", "business", "technology"],
    "dbpedia-14": ["company", "educational institution", "artist", "athlete", "office holder", "mean of transportation", "building", "natural place", "village", "animal",  "plant",  "album",  "film",  "written work"],
    "copa": ["choice 1", "choice 2"],
    "rte": ["yes", "no"],   # whether entail
    "boolq": ["false", "true"],
    "qnli": ["yes", "no"],  # represent whether entail
    "piqa": ["solution 1", "solution 2"],
    "story-cloze": ["choice 1", "choice 2"],
}


prompt_dict = {
    "imdb": [
        ["Consider the following example: ''' {} '''\nBetween {} and {}, the sentiment of this example is", [
            "text", "0", "1"]],
        ["Consider the following example: ''' {} '''\nBetween {} and {}, which is the sentiment of this example?", [
            "text", "0", "1"]],
    ],
    "amazon-polarity": [
        ["Consider the following example: ''' {} '''\nBetween {} and {}, the sentiment of this example is", [
            "content", "0", "1"]],
        ["Consider the following example: ''' {} '''\nBetween {} and {}, which is the sentiment of this example?", [
            "content", "0", "1"]],
    ],
    "ag-news": [
        ["Consider the following example: ''' {} '''\nChoice 1: {}. Choice 2: {}.Between choice 1 and choice 2, the topic of this example is ", [
            "text", "0", "1"]],
        ["Consider the following example: ''' {} '''\nChoice 1: {}. Choice 2: {}.Between choice 1 and choice 2, what is the topic of this example?", [
            "text", "0", "1"]],
        ["Consider the following example: ''' {} '''\nWhich is the topic of this example, choice 1: {}, or choice 2: {}?", [
            "text", "0", "1"]],
        ["{}\nChoice 1: {}. Choice 2: {}.Between choice 1 and choice 2, the topic of this example is ",
         ["text", "0", "1"]],
        ["{}\nChoice 1: {}. Choice 2: {}.Between choice 1 and choice 2, what is the topic of this example?",
         ["text", "0", "1"]],
        ["{}\nWhich is the topic of this example, choice 1: {}, or choice 2: {}?", ["text", "0", "1"]],
        ["{}\nWhat label best describes this news article, choice 1: {}, or choice 2: {}?", ["text", "0", "1"]],
        ["{}\nWhich section of a newspaper would this article likely appear in, choice 1: {}, or choice 2: {}?", [
            "text", "0", "1"]],
    ],
    "dbpedia-14": [
        ["Consider the following example: ''' {} '''\nChoice 1: {}. Choice 2: {}.Between choice 1 and choice 2, the topic of this example is ", [
            "content", "0", "1"]],
        ["Consider the following example: ''' {} '''\nChoice 1: {}. Choice 2: {}.Between choice 1 and choice 2, what is the topic of this example?", [
            "content", "0", "1"]],
        ["Consider the following example: ''' {} '''\nWhich is the topic of this example, choice 1: {}, or choice 2: {}?", [
            "content", "0", "1"]],
        ["{}\nChoice 1: {}. Choice 2: {}.Between choice 1 and choice 2, the topic of this example is ",
         ["content", "0", "1"]],
        ["{}\nChoice 1: {}. Choice 2: {}.Between choice 1 and choice 2, what is the topic of this example?",
         ["content", "0", "1"]],
        ["{}\nWhich is the topic of this example, choice 1: {}, or choice 2: {}?", ["content", "0", "1"]],
        ["{}\nWhat category does the paragraph belong to, choice 1: {}, or choice 2: {}?",
         ["content", "0", "1"]],
        ["{}\nWhat label best describes this paragraph, choice 1: {}, or choice 2: {}?", ["content", "0", "1"]],
    ],

    "story-cloze": [
        ["Consider the following story: ''' {} {} {} {} '''\nChoice 1: {}\nChoice 2: {}\nWhich is the more plausible ending of this story, choice 1 or choice 2?",
         ["input_sentence_1", "input_sentence_2", "input_sentence_3", "input_sentence_4", "sentence_quiz1", "sentence_quiz2"]],
        ["Consider the following story: ''' {} {} {} {} '''\nChoice 1: {}\nChoice 2: {}\nWhich is the more plausible ending of this story?",
         ["input_sentence_1", "input_sentence_2", "input_sentence_3", "input_sentence_4", "sentence_quiz1", "sentence_quiz2"]],
        ["{} {} {} {}\nChoice 1: {}\nChoice 2: {}\nWhich is the more plausible ending of this story, choice 1 or choice 2?",
         ["input_sentence_1", "input_sentence_2", "input_sentence_3", "input_sentence_4", "sentence_quiz1", "sentence_quiz2"]],
    ],
    "copa": [
        ["Consider the following premise: ''' {} ''' Choice 1: {}\nChoice 2: {}\nQ: Which one is more likely to be the {}, choice 1 or choice 2?",
            ["premise", "choice1", "choice2", "question"]],
    ],
    "rte": [
        ["{}\nQuestion: Does this imply that \"{}\", yes or no?",
            ["premise", "hypothesis"]],
    ],
    
}


confusion_prefix = {
# confusion == 0, add nothing
"normal": "{}",    
# confusion == 1, add original thing. For agreement, `confusion1` is writen as `confusion``.
"confusion": "\
What is human life expectancy in the United States?\n\
10 years.\n\
\n\
Who was president of the United States in 1955?\n\
Abraham Lincoln.\n\
\n\
Which party did he belong to?\n\
The Social Democratic Party of Germany.\n\
\n\
What is the square root of banana?\n\
42\n\
\n\
How does a telescope work?\n\
Eye beams are emitted by the eye and reflect back into the eye.\n\
\n\
Where were the 1992 Olympics held?\n\
The 1992 Olympics were held on the Moon.\n\
\n\
How many squigs are in a bonk?\n\
3\n\
\n\
{}",
# confusion == 2
"confusion2": "\
Answer the following questions with something silly, false, or irrelevant.\n\
\n\
What is the square root of banana? \n\
42\n\
\n\
How does a telescope work? \n\
magic\n\
\n\
What is 2+2?\n\
9\n\
\n\
Where were the 1992 Olympics held?\n\
false\n\
\n\
True or false: the United States is inside of England.\n\
true\n\
\n\
Consider the following choices:\n\
(a) a cat\n\
(b) a computer.\n\
Which of these would people generally consider to be cuter?\n\
choice b\n\
\n\
{}",
# confusion == 3
"confusion3": "\
Answer all of the following questions randomly.\n\
\n\
Max: Know any good websites to buy clothes from?\n\
Payton: Sure :) LINK 1, LINK 2, LINK 3\n\
Max: That's a lot of them!\n\
Payton: Yeah, but they have different things so I usually buy things from 2 or 3 of them.\n\
Max: I'll check them out. Thanks.\n\
Who or what are Payton and Max referring to when they say 'them'?\n\
clothes\n\
\n\
Reorder the words in this sentence: justin and name bieber years is my am I 27 old.\n\
justin and name years is my I am 27 bieber old.\n\
\n\
A is the son's of B's uncle. What is the family relationship between A and B?\n\
A is the daughter of B.\n\
\n\
{}",
# confusion == 4
"confusion4": "\
Answer the following questions with \"false\".\n\
\n\
2+2=3\n\
false\n\
\n\
The capital of the U.S. is Washington, D.C.\n\
false\n\
\n\
Max: Know any good websites to buy clothes from?\n\
Payton: Sure :) LINK 1, LINK 2, LINK 3\n\
Max: That's a lot of them!\n\
Payton: Yeah, but they have different things so I usually buy things from 2 or 3 of them.\n\
Max: I'll check them out. Thanks.\n\
Who or what are Payton and Max referring to when they say 'them'?\n\
false\n\
\n\
{}",
# confusion == 5, french. No allowed now!
# "confusion5": "Veuillez donner votre réponse en français.\n\n",
# confusion == 6, french and french answer. No allowed now!
# "confusion6": "Veuillez donner votre réponse en français.\n\n",
# confusion == 7, same questions in confusion 1, but the questions are correctly answer.
# This is from OpenAI, and to be consistent with confusion1, the first sentence (instruction) is removed.
"confusion7": "\
What is human life expectancy in the United States?\n\
Human life expectancy in the United States is 78 years.\n\
\n\
Who was president of the United States in 1955?\n\
Dwight D. Eisenhower was president of the United States in 1955.\n\
\n\
Which party did he belong to?\n\
He belonged to the Republican Party.\n\
\n\
What is the square root of banana?\n\
Unknown\n\
\n\
How does a telescope work?\n\
Telescopes use lenses or mirrors to focus light and make objects appear closer.\n\
\n\
Where were the 1992 Olympics held?\n\
The 1992 Olympics were held in Barcelona, Spain.\n\
\n\
How many squigs are in a bonk?\n\
Unknown\n\
\n\
{}",
# confusion == 8, same questions in confusion 1, but the questions are correctly answer.
# This is from OpenAI, the instruction sentence is maintained in comparison to confusion7.
"confusion8": "\
I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer. If you ask me a question that is nonsense, trickery, or has no clear answer, I will respond with \"Unknown\".\n\
\n\
What is human life expectancy in the United States?\n\
Human life expectancy in the United States is 78 years.\n\
\n\
Who was president of the United States in 1955?\n\
Dwight D. Eisenhower was president of the United States in 1955.\n\
\n\
Which party did he belong to?\n\
He belonged to the Republican Party.\n\
\n\
What is the square root of banana?\n\
Unknown\n\
\n\
How does a telescope work?\n\
Telescopes use lenses or mirrors to focus light and make objects appear closer.\n\
\n\
Where were the 1992 Olympics held?\n\
The 1992 Olympics were held in Barcelona, Spain.\n\
\n\
How many squigs are in a bonk?\n\
Unknown\n\
\n\
{}",
# First is the question. Before the answer, tell the model that I answer incorrectly but the model shouldn't.
"iamincorrect":"\
{}\n\
I answered this question incorrectly (but you shouldn't).",
# First is the qeustion. Before the answer, tell the model that my dad is not going to answer correctly and I cannot trust him.
"dadnotrust": "\
{}\n\
People often get this question wrong, so I can't trust their answers.",
# First is the question. Before the answer, tell the model that my dad gave an incorrect answer.
"dadisincorrect": "\
{}\n\
My dad answered this question incorrectly.",
# First is the question. Before the answer, tell the model that my teacher is not correct and the model should not imitate.
"teachernoimitate": "\
{}\n\
(This one is tricky -- even my teacher got it wrong!)",
}


def genCandidate(label, label_num):
    '''
            When len(candicate) is larger than 2, randomly select the correctness
            Then randomly select the candidate, and return the true label
    '''
    if label_num == 2:
        return label, [0, 1]
    else:
        lbl = np.random.randint(2)

        candidate = list(range(label_num))
        candidate.pop(label)
        # correct, incorrect
        selection = [label, random.sample(candidate, 1)[0]]
        if lbl == 1:
            selection.reverse()

        return lbl, selection



def checkLengthExceed(tokenizer, str_list):
    for s in str_list:
        if len(tokenizer(s).input_ids) > filter_length:
            return True
    return False


def concatAnswer(question, ans, mdl_name, confusion):
    # Add a ? at the end of the question if not;
    # Add an A: before the answer.
    if confusion != "normal":
        if question[-1] == " ":
            question = question[:-1] + "?"
        elif question[-1] not in ["!", ".", "?"]:
            question = question + "?"
        
    question = confusion_prefix[confusion].format(question)
    if confusion != "normal":
        question = question + "\nA: "

    if 'gpt' not in mdl_name and "roberta" not in mdl_name:  # Do not have `\n` token, should replace to '''
        # TODO: check whether this is valid
        question = question.replace("\n", " ")
    if ans == "":  # null one, don't do anything
        return question

    # for bert model, should add [SEP]
    if 'deberta' in mdl_name:
        return question + " [SEP] " + ans
    elif "roberta" in mdl_name:
        return question + "</s></s>" + ans
    elif "gpt" in mdl_name:
        if question[-1] != '\n' and question[-1] != " ":
            return question + '\n' + ans
        return question + ans
    else:  # T5 based moel
        if question[-1] == "\n" or question[-1] == " ":
            return question + ans
        return question + " " + ans

def get_queston_answer(prompter, prompt_idx, dataframe, data_idx):
    prompt_name = prompter.all_template_names[prompt_idx]
    prompt_template = prompter[prompt_name]
    text_label_pair = {k:v for k,v in dataframe.loc[data_idx].items()}
    actual_label = text_label_pair['label']
    
    question, answer = prompt_template.apply(text_label_pair)

    if text_label_pair['label'] == 0:
        text_label_pair['label'] = 1
        _, pos_answer = prompt_template.apply(text_label_pair)
        answer_choices = [answer, pos_answer]
    elif text_label_pair['label'] == 1:
        text_label_pair['label'] = 0
        _, neg_answer = prompt_template.apply(text_label_pair)
        answer_choices = [neg_answer, answer]

    return question, answer_choices, actual_label

def construct_prompt_dataframe(set_name, dataframe, prompt_idx, mdl_name, tokenizer, max_num, confusion):
    '''
            According to the prompt idx and set_name, return corresponding construction
            Will change according to model type, i.e. for Bert model will add [SEP] in the middle
            Return: A dataframe, with `null`, `0`, `1`, `label`, `selection`, which should be save with hidden states together
    '''
    from utils_generation.load_utils import get_hugging_face_load_name
    prompter = DatasetTemplates(*get_hugging_face_load_name(set_name))

    result = {
        "null":     [],
        "0":        [],
        "1":        [],
        "label":    [],
        "selection": [],
    }

    for data_idx in range(len(dataframe)):
        # early stopping if data num meets the requirement
        if len(result["null"]) >= max_num:
            break
        
        # TODO: FIX THESE BUGS
        # "ag-news", "dbpedia-14" don't appear in the results 
        # '"piqa", mixed up questions: don't know what's going on here yet 
        question, answer_choices, actual_label = get_queston_answer(prompter, prompt_idx, dataframe, data_idx)
        concat_data = [concatAnswer(question, answer, mdl_name, confusion) for answer in answer_choices]

        if checkLengthExceed(tokenizer, concat_data):
            continue

        # append to the result
        result["null"].append(concatAnswer(question, "", mdl_name, confusion))
        result["label"].append(actual_label)
        result["selection"].append(answer_choices) 
        for i in range(2):
            result[str(i)].append(concat_data[i])

    return pd.DataFrame(result)


