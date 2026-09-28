ML Challenge 2026: Business Entity Resolution Solution 
Submission Date: 27 September 2026
1.	Executive Summary 
We developed a hybrid business entity resolution pipeline combining country-aware blocking, normalized business name and address matching, supervised machine learning, and precision-oriented threshold selection. The solution uses a HistGradientBoosting classifier trained on labelled entity pairs with name and address similarity features, while candidate generation substantially reduces the comparison space before ML scoring.
The approach is designed for the challenge's one-to-many matching setting, where each Source 1 entity can correspond to zero, one, or multiple Source 2 or Source 3 entities.
2.	Methodology 
2.1 Problem Analysis
Exploratory analysis showed that the dataset contains substantial variation and repetition in business names and addresses.
Source 1 contains 2,206,821 entities in the training data. Source 2 contains 5,034,616 entities and Source 3 contains 5,285,603 entities.
Key observations from the training data include:
Source 1 has 1,539,229 unique business names, with repeated names occurring across 667,592 rows.
Source 1 has 2,130,606 unique addresses, with repeated addresses occurring across 76,215 rows.
Source 2 contains 168,967 empty addresses.
Source 3 contains 175,916 empty addresses.
Source 1 contains no empty business names, addresses, or countries in the analysed training data.
Source 2 and Source 3 contain no empty business names or countries.
Business names and addresses contain formatting variation, punctuation differences, abbreviation differences, and token-order variation.
The training ground truth contains one-to-many relationships, with a maximum of 11 matched target entities for a Source 1 entity.
The training ground truth contains 7,638,365 total matching links across 2,206,821 Source 1 entities.
Country agreement was observed for all 7,638,365 labelled matching links in the training ground truth.
The evaluation is precision-oriented because the challenge uses macro F0.5, making false positive merges particularly important.
2.2 Solution Strategy
Approach Type: Blocking + Supervised Classifier / Hybrid
Core Innovation: The solution combines deterministic candidate generation with a supervised pairwise matching model. Candidate pairs are first generated using country-aware normalized business name and address blocking. The resulting candidate pairs are then scored using a machine learning classifier using multiple name and address similarity features.
The pipeline consists of the following stages:
1.	Data loading and validation. 
2.	Text normalization. 
3.	Country-aware candidate generation. 
4.	Feature extraction for candidate entity pairs. 
5.	Supervised classification using labelled training pairs. 
6.	Validation-based probability threshold selection. 
7.	ML scoring of the complete test candidate set. 
8.	Generation of candidate_pairs.tsv and matching_results.tsv. 
9.	Submission validation. 
10.	Candidate Generation (Blocking) 
Candidate generation is used to avoid comparing every Source 1 entity against every Source 2 and Source 3 entity.
The blocking strategy uses normalized business names and normalized addresses within the same country.
Blocking keys used:
Country combined with normalized business name.
Country combined with normalized business address.
Text normalization includes lowercasing, Unicode normalization, removal of accent marks, punctuation normalization, whitespace normalization, and business-name abbreviation normalization.
Common business-name abbreviations such as ltd, co, corp, inc, pvt, and llc are normalized to their expanded forms where applicable.
Candidate pairs generated: approximately 16.1 million candidate links on the test data.
The current test candidate set contains approximately 9.3 candidates per Source 1 entity on average.
How true matches were protected:
The blocking strategy uses the union of normalized-name and normalized-address candidates rather than requiring both fields to match simultaneously. This allows a candidate to enter the candidate set when either of these strong identifying signals agrees within the same country.
The candidate set is retained before the final ML filtering stage so that the classifier operates on the generated candidate pairs.
4.	Matching Model 
Features used:
Name features:
Exact normalized business name match.
RapidFuzz name similarity ratio.
Token-sort similarity.
Token-set similarity.
Token Jaccard similarity.
Address features:
Exact normalized address match.
RapidFuzz address similarity ratio.
Token-sort similarity.
Token-set similarity.
Address Jaccard similarity.
Other features:
Country exact-match indicator.
A total of 11 features are used by the trained classifier.
Model type:
HistGradientBoostingClassifier.
The model is trained using labelled positive and negative entity pairs derived from the supplied training ground truth.
The training process uses a sampled subset of Source 1 entities and retrieves the corresponding labelled target records. Positive examples are taken from the ground-truth matching relationships, while negative examples are generated from non-matching blocked candidate pairs.
The trained model is stored as:
output/entity_matcher.joblib
Threshold selection method:
The classifier produces a probability for each candidate pair. A validation-based threshold search was performed over multiple probability thresholds.
The selected threshold from the current training experiment is:
0.75
The sampled training validation experiment produced:
Validation F0.5: 0.994935
This value is a model-development validation result and is not the final official challenge score.
5.	Results and Error Analysis 
F0.5 Score (macro): 0.994935 on the current sampled training validation experiment.
Selected ML threshold: 0.75.
Training positive pairs: 100,000.
Training negative pairs: 31,525.
Total supervised training pairs: 131,525.
Number of ML features: 11.
Common false positives:
Potential false positives are expected primarily when multiple businesses share highly similar or identical normalized names or addresses. Repeated business names can produce multiple candidates for the same Source 1 entity.
The classifier therefore uses both name and address similarity rather than relying solely on exact business-name matching.
Common false negatives:
Potential false negatives can occur when a true matching business has substantial variation in both its business name and address and therefore does not enter the blocking candidate set, or when the ML probability falls below the selected threshold.
The current pipeline therefore uses multiple fuzzy similarity features to reduce the risk of incorrect merges.
Note:
Detailed final test-set false-positive and false-negative analysis will be completed after the final ML inference and validation run.
6.	Conclusion 
The MICRON solution implements a scalable hybrid entity-resolution pipeline using text normalization, country-aware blocking, fuzzy similarity features, and supervised machine learning. The approach is designed to handle the challenge's one-to-many matching structure while emphasizing precision through F0.5-oriented threshold selection.
The current development results demonstrate strong performance on the sampled training validation set. The final test outputs will be validated using the official challenge submission validator before submission.
Appendix
A. Code Artefacts
The complete runnable implementation is organized under:
code/business_entity_resolution/
The main source components include:
config.py
Contains project paths, dataset locations, output locations, and configuration settings.
data_loader.py
Contains utilities for loading entity and ground-truth TSV files.
normalization.py
Contains business name, address, and country normalization functions.
fast_matcher.py
Generates candidate entity pairs using deterministic country-aware blocking based on normalized business names and addresses.
ml_matcher.py
Trains the supervised machine learning model and selects the probability threshold.
apply_model.py
Applies the trained machine learning model to the test candidate pairs and generates the final matching results.
validate_submission.py
Validates the submission files for required structure, coverage, duplicate restrictions, candidate consistency, and identifier validity.
The trained model is stored as:
output/entity_matcher.joblib
The final pipeline produces:
output/candidate_pairs.tsv
and
output/matching_results.tsv
The candidate generation stage produces the candidate set before final ML scoring, while the matching results file contains the final ML-selected matches.
B. Additional Results
Training Dataset Statistics
Train Source 1: 2,206,821 rows
Train Source 2: 5,034,616 rows
Train Source 3: 5,285,603 rows
Training ground-truth links: 7,638,365
Test Dataset Statistics
Test Source 1: 1,732,544 rows
Test Source 2: 4,887,273 rows
Test Source 3: 5,082,316 rows
Combined test targets: 9,969,589
Ground Truth Match Distribution
Source 1 entities with no match: 123,247
Source 1 entities with exactly 1 match: 119,157
Source 1 entities with exactly 2 matches: 375,212
Source 1 entities with exactly 3 matches: 530,841
Source 1 entities with exactly 4 matches: 484,115
Source 1 entities with exactly 5 matches: 321,957
Source 1 entities with 6 or more matches: 252,292
Maximum number of matches for a Source 1 entity: 11
Model Development Results
Sampled Source 1 entities: 80,000
Unique positive target IDs used: 276,068
Positive training pairs: 100,000
Negative training pairs: 31,525
Total training pairs: 131,525
Number of ML features: 11
Machine learning model: HistGradientBoostingClassifier
Selected probability threshold: 0.75
Validation F0.5: 0.994935
Final test-set results and official validation results will be added after ML inference is completed.

