# CbEL: Confidence-based Entity Linking for Historical Medical Texts

Idea: Using large language model to link entities with confidence estimation.

## Pipeline
```python
FUNCTION EntityLink(text):
    # Stage 0: Detect & Normalize mentions
    mentions = NER(text)
    FOR m IN mentions:
        m.normalized = LLM("Normalize mention given context", m)
    
    # Stage 1: Map-Reduce Candidate Ranking (MREL)
    FOR m IN mentions:
        candidates = KB_Lookup(m.normalized)
        chunks = Split_Into_Overlapping_Batches(candidates)
        
        votes = {}  # QID → score
        FOR chunk IN chunks:
            selected = LLM("Pick relevant IDs from chunk given context", chunk)
            FOR qid IN selected:
                votes[qid] += 1/|selected|  # Higher score if selected alone
        
        m.candidates = Top_K_By_Score(votes)
    
    # Stage 2: Confidence-Based Iteration (CBI)
    confirmed = []
    pending = mentions
    FOR iter IN 1..K:
        IF pending IS EMPTY: BREAK
        
        new_pending = []
        FOR m IN pending:
            # Enrich context with confirmed entity descriptions
            context = m.context + Format_Descriptions(confirmed)
            
            result = LLM("Link mention to candidates with confidence", 
                        context, m.candidates)
            
            IF result.confidence > THRESHOLD AND CoinFlip(result.confidence):
                confirmed.ADD(m WITH result.entity)
            ELSE:
                new_pending.ADD(m)  # Retry next iteration
        
        pending = new_pending
    
    RETURN confirmed + pending WITH sufficient_confidence
```

This implementation is based on the [ELEVANT project](https://github.com/ad-freiburg/elevant/tree/master), you can find our implementation of CbEL in `linkers/cbel.py`. This is implemented for general entity linking tasks. To make this pipeline work on other domains, change the prompt and NER models.

You can copy and paste `linkers/cbel.py` to `./src/elevant/linkers/cbel.py` and `./llm_client.py` to `./src/elevant/llm_client.py` in your ELEVANT project. After that, register new linker to the project and you can run it on ELEVANT benchmarks.

## Dataset
You can find the MedHistory benchmark on [HuggingFace](https://huggingface.co/datasets/npvinHnivqn/MedHistory). This benchmark is collected and annotated from 20th century BMJ papers.

