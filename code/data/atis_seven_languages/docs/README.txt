# ATIS NLU - Seven Languages

This dataset contains human translation and annotation of the original English
ATIS corpus [1] into six new languages: Spanish, German, French, Portuguese, 
Chinese, Japanese. The training and test split in each language is the one used
in [2]. Please cite [3] when referring to the dataset.


## Authors
Saab Mansour, Batool Haider


## Languages
English, Spanish, German, French, Portuguese, Chinese, Japanese

Each language has 4978 utterances for training and 893 utterances for testing.


## Applications
The ATIS (Air Travel Information Services) collection was developed to support 
the research and development of speech understanding systems. The release of 
the 7-way parallel ATIS corpus is intended to foster research into multilingual
and cross-lingual NLU methods. 


## Format
For each language, the dataset has 2 files:
train_$language.tsv
test_$language.tsv

Where $language can be one of English (EN), Spanish (ES), German (DE), 
French (FR), Portuguese (PT), Chinese (ZH) and Japanese (JA)

The training set contains 4978 utterances selected from the Class A
(context independent) training data in the ATIS-2 and ATIS-3 corpora,
while the test set contains 893 utterances from the ATIS-3 Nov93 and
Dec94 datasets. Each utterance has its named entities marked via table
lookup, including domain specific entities such as city, airline,
airport names, and dates.

Each line in the tsv files includes the following 4 tab-separated columns 
corresponding to an utterance:
- utterance id
- utterance text
- slot labels (BIO format)
- intent label


## Collection procedure
For each target language translation, we hire professional native translators 
to translate the English utterances into the target language and annotate the
slots at the same time. When translating, the translators are required to 
preserve the meaning and structure of the original English sentences as much 
as possible including noise such as repetitions that are common in the spoken 
language. To get the slot labels for the translated utterances, we ask the 
translators to project the slot labels from the English segments to the 
corresponding translated segments in the target language.

To guarantee the quality of the annotated data, we ask a third party to perform
several rounds of qualification checks until no issues are reported.


Contact
=======

For questions about the dataset, please email saabm@amazon.com

References
==========

[1] LDC93S5 ATIS2, LDC94S19 ATIS3 Training Data, LDC95S26 ATIS3 Test Data
[2] Shyam Upadhyay, Manaal Faruqui, Gokhan Tur, Dilek Hakkani-Tur, Larry Heck. (Almost) Zero-Shot Cross-Lingual Spoken Language Understanding. IEEE ICASSP 2018.
[3] Weijia Xu, Batool Haider, Saab Mansour. End-to-End Slot Alignment and Recognition for Cross-Lingual NLU. EMNLP 2020.
