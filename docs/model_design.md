# Model Design

The first version implements one main architecture only: local multi-scale Graph-SPIB.

Local graphs are not full-system proxies. They are local dynamic sampling units centered on PE or PP segments and are used to learn mobility, packing, free-volume, PE/PP contact, and local relaxation signals from short trajectory windows.

System-level transport properties are predicted only after grouping many local embeddings by `system_id` and applying distribution pooling. The pooled representation includes latent statistics, PE/PP-centered statistics, interface statistics, local composition histograms, contact fractions, state fractions, and global condition variables.

The project deliberately does not implement a descriptor-only main model, does not use a full simulation-box raw graph, does not use PySR or SISSO, does not use PyG advanced sampling, and does not load polyBERT, TransPolymer, or pretrained Graph-SPIB weights.
