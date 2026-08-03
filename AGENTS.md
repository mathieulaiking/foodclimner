# SafeFood4ClimDiet - A Corpus for Food practices related to climate change

Zero-shot NER with encoders and LLMS on manually annotated corpus for food practices related to climate change. 

## Quick start
- Package manager: `uv` (Python ≥3.12, `.python-version` = 3.14)
- Install: `uv pip install -e .`
- Test: `uv run pytest`
- Lint: `uv run ruff check`

## Format choices : 

### Annotations

We use [brat standoff format](https://brat.nlplab.org/standoff.html), the annotations are in data/brat-sf4cd-annots. There are different versions specified in the changelog.md file.

### NER-Annotated document 

We want to have a standard format that takes in account nested entities (mentions that have spans which have overlapping spans, that includes intersection, inclusion, equality, with different types overlapping) as well as discontinuous entities (a discontinuous entity is a single entity that have spans that have "holes" inside, so multiple sub-spans that are not consequent). We also want to be tokenization-independant so we keep the text as-is without tokenization and use character offsets. We choose the following format , for a given document identified by a `document_id` containing entities with n sub-spans : 

```json
{
    "id" : "document_id",
    "text": "[Document text containing entities]",
    "entities" : [
        {
            "type":"[entity_type]",
            "text": "[entity_text]",
            "offsets" : [
                [start_offset_1,end_offset_1],
                ...
                [start_offset_n, end_offset_n]
            ]
        },
        ...
    ]
}
```

### Entities Types Informations

For the NER task, depending on the approaches we take, we may have to enrich the entity type with other informations (especially with LLMs) , so we decide to use the following format. Depending on the method used, we can use or not the different attributes of theses entities, the only required one being the name.

```json
{
    "name": "[entity_class_name_1]",
    "definition": "[entity_definition]",
    "examples": [
        "[example_1]",
        "...",
        "[example_n]"
    ],
    "counter_examples" : [
        "[counter_example_1]",  
        "...",
        "[counter_example_n]"
    ]
}
```

### Datasets 

Datasets used for experiments should be in a standardized format. We defined a dataset as a dataset directory containing : 
* a **test.jsonl** file, which contains "Document entities" in the given format above (one per line, jsonlines format)
* a **entities.json** which contains a list of the different entities types in the "Entities Information" format (basic json format)

So experiments runner files should always define a `--data_dir` argument.
 

## Baselines output formats and parsing

### GLiNER

We use [GLiNER](https://urchade.github.io/GLiNER/) , which is a bert like encoder. It is trained to assign an entity to each token given the representation of the entity type name only.

For advanced usage see [this link](https://urchade.github.io/GLiNER/usage.html).

#### GLiNER Output Format

```json
{
    'start': 0,
    'end': 10,
    'text': 'John Smith',
    'label': 'person',
    'score': 0.95
}
```

#### Type of NER handled by GliNER

This format does not handle the discontinuous entities, as GLiNER does not handle it. However GliNER can handle Nested NER : 

```python
from gliner import GLiNER

model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
text = "The University of California, Berkeley is located in California."
labels = ["university", "location"]

# Flat NER: No overlapping entities (default)
entities_flat = model.predict_entities(text, labels, flat_ner=True)
print("Flat NER:", [e['text'] for e in entities_flat])
# Output: ['University of California, Berkeley', 'California']

# Nested NER: Allow overlapping entities
entities_nested = model.predict_entities(text, labels, flat_ner=False)
print("Nested NER:", [e['text'] for e in entities_nested])
# Output: ['University of California, Berkeley', 'California, Berkeley', 'California']
```

#### Efficiency

For processing multiple texts efficiently, use the inference method:
```python
from gliner import GLiNER

model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")

# Multiple texts to process
texts = [
    "Apple Inc. was founded by Steve Jobs in Cupertino, California.",
    "Google LLC is headquartered in Mountain View.",
    "Amazon was started by Jeff Bezos in Seattle."
]

labels = ["organization", "person", "location"]

# Process all texts at once
all_entities = model.inference(texts, labels, batch_size=3, threshold=0.5)

# Display results for each text
for i, entities in enumerate(all_entities):
    print(f"\nText {i+1}: {texts[i]}")
    print("Entities:")
    for entity in entities:
        print(f"  - {entity['text']} ({entity['label']}): {entity['score']:.2f}")
``` 

#### Local mode

```python
from gliner import GLiNER

# Load from local directory
model = GLiNER.from_pretrained("/path/to/local/model")

# Or load from HuggingFace Hub with local cache
model = GLiNER.from_pretrained(
    "urchade/gliner_small-v2.1",
    cache_dir="./model_cache"  # Cache models locally
)
```

### GoLLIE


## Experiments 

For the experiments (src.experiment) , we always want to save the results to a defined directory with the following structure : 

├── [pred_dir_name]
│   ├── raw_preds (directory containing predictions file for each input as .txt or .json files)
│   └── predictions.jsonl (jsonlines file containing parsed predictions to the NER-Annotated format)
│   └── config.json (config file, arguments given to the experiment file, model config, ... everything interesting to track)
│   └── stats.json (file containing efficiency, experiments should track the total time, inference time, time per example)

Experiments should have a `--debug` mode that will just run the whole pipeline on a few examples (10 examples) to check that it runs well

## Evaluation

The evaluation is done at the entity-level, not the token-level. We measure precision, recall and f1-score. We perform micro-average over the elements of the dataset. The main metric is **micro-f1-score**.

We compare entities in the format given in the "NER-annotated document" section. 

There should be different modes of evaluation :
* strict mode : exact boundary surface string match and exact entity type
* exact mode : exact boundary match regardless of entity type
* partial mode : partial boundary match over the surface string + exact entity type 

## Jobs launchers

After defining everything, jobs should run on a cluster. You should create bash scripts that will launch multiple experiments on different nodes. There already is a `template.slurm` file that fits the Jean-Zay cluster. However when using the agent we will not be on jean-zay cluster so don't try to run the scripts to launch jobs. We should define two lists in these files : 
* models list that will be lists of tuple containing : model name, gpu type on which to run it, gpu number 
* datasets list that will be a lists of strings containing dataset paths
These two lists will be edited by the user (by commenting / adding elements) according to his needs.

Jean-zay cluster has different GPUs accessible : GPU partitions at [this link](http://www.idris.fr/docs/jean-zay/slurm/slurm_partitions_gpu), we have access currently to V100, A100, and H100. 

* IMPORTANT : This cluter nodes do not have access to internet, so when creating scripts to run on jean-zay ensure to use a mode using local models
* memory allocation at [this link](http://www.idris.fr/docs/jean-zay/slurm/exec_alloc-mem)

## Coding rules

We want to follow Object-oriented Programming good practices , as well as using pythonic code, so using well the python standard library. Also using carefully chosen design patterns to keep the code simple, maintainable, and understandable. Simplicity should always be the main component.

Use abstract classes to avoid code duplication. Carefully choose their naming and responsibilty (concrete class = class that will be directly used ; abstract class will be for abstracting some parts, you can also use interfaces or other OOP principles if needed).

For python use :
* **argparse** should always be used for files that needs to be run, also the arguments definition and parsing should be packed into a `parse_args` function that should be on the top of the file (after the imports and optionnally some constants/Enums needed in the function). This function parse the different arguments, verify there validity and throws errors if some are wrong, and then returns the parsed args. 
* use @dataclasses decorator and create objects structures for every non standard types of objects that we'll define so that it is clearer what each function returns, which structure.
* use python typing (arguments, function returns)

Always document the functions that can be called and used in the codebase elsewhere (behaviour, arguments, returns).

## Naming conventions

* Entities types (classes) : should be snake_cased, so we should define a normalization file to handle conversion from multiple possible format (CamelCase, type with spaces, hyphens, ...) it should be the same across the two datasets files (test.jsonl and entities.json) as well as in the predictions file (so the parsing module should implement this)
* CLI files arguments : should follow bash arguments (lowercase, with a '-' separating words) and use abbreviations for words, also adding as much single letter flags for shorter script calls


