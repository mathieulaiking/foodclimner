# NER Evaluation Nested Discontinuous

## ChatGPT

For **Named Entity Recognition (NER)** with **nested entities** and **discontinuous entities**, evaluation becomes more subtle than standard flat BIO tagging. Since you asked for:

- **entity-level evaluation**
- **strict matching**
- **exact mention boundaries**
- **exact entity type**
- no token-level scoring

the clean way to think about evaluation is:

> Convert both gold annotations and predictions into sets of **entity objects**, then compare those sets exactly.

The main challenge is how entities are represented.

---

# 1. What changes compared to standard flat NER?

In ordinary flat NER:

> Each entity = `(start, end, label)`

Example:

Text:
> Barack Obama visited Paris

Gold:

```
[(0,2, PERSON),
 (3,4, LOCATION)]
```

Prediction:

```
[(0,2, PERSON),
 (3,4, LOCATION)]
```

Evaluation is straightforward.

But flat spans break when entities:

1. **overlap (nested)**
2. **contain multiple separated spans (discontinuous)**

So annotation representation matters.

---

# 2. Nested Entities

## Definition

A nested entity is an entity fully or partially contained inside another.

Example:

Text:

> [University of [California]LOC]ORG

Gold:

```
ORG: "University of California"
LOC: "California"
```

The spans overlap.

---

## Why BIO fails

Standard BIO assumes:

- one label per token
- entities cannot overlap

Example:

| Token | BIO |
|--------|-----|
| University | B-ORG |
| of | I-ORG |
| California | I-ORG |

No place to also encode:

```
California → LOC
```

---

# Annotation schemes adapted for nested NER

Several schemes exist.

---

## A. Span-based annotation (most evaluation-friendly)

Represent entities independently.

Format:

```
(start, end, type)
```

Example:

Text:
> New York University

Tokens:

```
0 New
1 York
2 University
```

Annotations:

```
(0,2, ORG)
(0,1, LOC)
```

Meaning:

```
ORG → New York University
LOC → New York
```

### Why adapted?

Because:

- overlap allowed
- no token conflicts
- evaluation becomes set comparison

---

## B. Layered BIO / stacked BIO

Example:

Layer 1:

```
B-ORG I-ORG I-ORG
```

Layer 2:

```
B-LOC I-LOC O
```

Useful for sequence labeling models.

But evaluation usually converts back to spans.

---

## C. Hypergraph / region representations

Used in re systems.

Entity:

```
node(start,end,label)
```

Allows arbitrary overlap.

Good for modeling.

Evaluation still becomes span comparison.

---

# Strict entity-level evaluation for nested entities

Represent:

```
Entity =
(start,end,type)
```

Prediction is correct iff:

```
start_pred == start_gold
AND
end_pred == end_gold
AND
type_pred == type_gold
```

---

## Example 1 — exact match

Text:

> Bank of America headquarters

Gold:

```
(0,2, ORG)
```

Prediction:

```
(0,2, ORG)
```

Result:

```
TP = 1
FP = 0
FN = 0
```

Perfect.

---

## Example 2 — boundary error

Gold:

```
(0,2, ORG)
```

Prediction:

```
(0,1, ORG)
```

Result:

```
TP = 0
FP = 1
FN = 1
```

Even though overlap exists.

Strict evaluation gives no credit.

---

## Example 3 — type error

Gold:

```
(0,2, ORG)
```

Prediction:

```
(0,2, LOC)
```

Result:

```
TP=0
FP=1
FN=1
```

---

## Example 4 — missing nested entity

Gold:

```
(0,2, ORG)
(2,2, LOC)
```

Prediction:

```
(0,2, ORG)
```

Result:

```
TP=1
FN=1
FP=0
```

Recall drops.

---

## Metrics

Compute globally:



$$Precision=\frac{TP}{TP+FP}$$





$$Recall=\frac{TP}{TP+FN}$$





$$F1=\frac{2PR}{P+R}$$



where entities are exact span-label tuples.

---

# 3. Discontinuous Entities

## Definition

A discontinuous entity consists of multiple separated text segments.

Example:

Text:

> He lives in New and works in York.

Suppose entity:

```
LOC = "New ... York"
```

Entity spans:

```
[(3,3),(8,8)]
```

Another realistic example:

> acute and chronic renal failure

Entity:

```
Disease:
[(0,0),(3,4)]
```

meaning:

```
acute renal failure
```

Tokens between spans are excluded.

---

## Why BIO fails

BIO assumes contiguous segments.

Example:

```
acute and chronic renal failure
```

Cannot express:

```
acute ____ renal failure
```

without special conventions.

---

# Annotation schemes adapted for discontinuous entities

---

## A. Multi-span entity representation (recommended)

Represent:

```
([span1, span2, ...], label)
```

Example:

```
([(0,0),(3,4)], DISEASE)
```

### Why adapted?

Because:

- exact mention structure preserved
- unlimited gaps
- evaluation straightforward

---

## B. Extended BIO (BIOHD, BIOES+, etc.)

Example:

```
B-Disease
O
B-Disease
I-Disease
```

plus linking markers.

Problem:

- decoding complexity
- difficult reconstruction

Evaluation still occurs on reconstructed spans.

---

## C. Graph annotation

Entity:

```
E:
 span1
 span2
 edges
 label
```

Useful for complex biomedical NER.

---

# Strict entity-level evaluation for discontinuous entities

Represent:

```
Entity =
([s1,e1],[s2,e2],...,type)
```

Correct iff:

- same number of segments
- each segment boundaries match
- same ordering
- same label

Formally:



$$Pred = Gold$$



exactly.

---

## Example 1 — exact

Gold:

```
([(0,0),(3,4)], DISEASE)
```

Prediction:

```
([(0,0),(3,4)], DISEASE)
```

TP=1

---

## Example 2 — missing segment

Gold:

```
([(0,0),(3,4)], DISEASE)
```

Prediction:

```
([(3,4)], DISEASE)
```

Result:

```
TP=0
FP=1
FN=1
```

No partial credit.

---

## Example 3 — wrong order

Gold:

```
[(0,0),(5,5)]
```

Prediction:

```
[(5,5),(0,0)]
```

If ordering is part of canonical representation:

```
incorrect
```

Most implementations normalize order first.

---

## Example 4 — merged discontinuous prediction

Text:

> acute and chronic renal failure

Gold:

```
[(0,0),(3,4)] DISEASE
[(2,4)] DISEASE
```

Prediction:

```
[(0,4)] DISEASE
```

Result:

```
TP=0
FP=1
FN=2
```

Strict evaluation penalizes merging.

---

# 4. Unified representation for nested + discontinuous NER

The most general representation is:



$$Entity=(\{(s_1,e_1),(s_2,e_2),...\}, label)$$



Examples:

Nested:

```
({(0,2)}, ORG)
({(2,2)}, LOC)
```

Discontinuous:

```
({(0,0),(3,4)}, DISEASE)
```

Nested + discontinuous:

```
({(0,1),(5,6)}, EVENT)
({(5,6)}, LOCATION)
```

Evaluation:

```
TP = |Gold ∩ Pred|
FP = |Pred − Gold|
FN = |Gold − Pred|
```

where equality means:

```
all spans identical
AND
same label
```

---

# 5. Typical error taxonomy under strict entity-level evaluation

| Error | Example | Counts |
|---|---|---|
| Boundary error | predicted shorter/longer | FP + FN |
| Type error | ORG instead of LOC | FP + FN |
| Missing nested entity | outer found, inner missed | FN |
| Hallucinated entity | predicted extra | FP |
| Segment omission | missing discontinuous part | FP + FN |
| Segment merge | merged multiple entities | FP + FN |
| Segment split | one entity → multiple | FP + FN |

---

# Practical recommendation

For **strict entity-level evaluation**:

1. **Convert annotations into entity objects**
 - Nested → `(start,end,type)`
 - Discontinuous → `([segments],type)`

2. **Canonicalize**
 - sort segments
 - normalize offsets

3. **Compare exact equality**

4. Compute:



$$P,R,F1$$



using **micro-averaging over entities**.

This evaluation is essentially what modern span-based NER benchmarks use because it remains valid even when entities are nested, overlapping, or discontinuous.

