#!/bin/bash
# v12 recipe — the commercial-clean mix (NC-free successor to v11c).
#
# Dropped vs v11c (NC-licensed / NC-derived, license audit 2026-08-23):
#   Leipzig che_community_2017/2023, kir_community_2017, tgk_community_2022,
#   uzb_community_2017 (CC-BY-NC-SA); the-cramer-project/Kyrgyz_News_Corpus
#   (CC-BY-NC-4.0); muhtasham/tajik-corpus (Leipzig-derived despite CC-BY tag).
# Glot500 configs row-filtered on their `dataset` source column
#   (Leipzig*|nllb_other_til|mtdata|Bible* = NC/restricted slices).
# Orthographic pollution gates (empirically validated vs trusted corpora):
#   tgk sources: no_chars=ўы  (ў = Uzbek-only; ы = Russian — Tajik has neither)
#   kir HPLT + its diverse pool: no_chars/pool_no_chars=ҕһ (Sakha-only letters)
#   uzn/che have NO orthographic Russian gate (alphabets overlap) — residual
#   label noise there is documented; a lower-threshold judge pass is the
#   un-pulled lever (rus is not in their twin groups).
# GoURMET (OPUS) was dropped: object.pouta.csc.fi unreachable from our
#   network — optional re-add (~23k ky pairs) if reachable.
# Run from train/:  uv sync && bash recipes/v12.sh
# (first run downloads everything; ~2.5GB MADLAD + HPLT/Glot500 streams)
set -euo pipefail
cd "$(dirname "$0")/.."
uv run spellman-train mix --out data/v12 \
  --source fineweb2:docs_per_lang=3600,per_doc=4,langs_exclude=tgk \
  --source tatoeba:train_per_lang=8000 \
  --source 'hf:repo=cis-lmu/Glot500,config=tat_Cyrl,exclude=dataset=Leipzig*|nllb_other_til|mtdata|Bible*,lang=tat,docs=3000,per_doc=4' \
  --source 'hf:repo=cis-lmu/Glot500,config=tgk_Cyrl,lang=tgk,exclude=dataset=Leipzig*|nllb_other_til|mtdata|Bible*,no_chars=ўы,docs=12000,per_doc=2' \
  --source 'hf:repo=cis-lmu/Glot500,config=sah_Cyrl,exclude=dataset=Leipzig*|nllb_other_til|mtdata|Bible*,lang=sah,docs=3000,per_doc=4' \
  --source 'hf:repo=cis-lmu/Glot500,config=udm_Cyrl,exclude=dataset=Leipzig*|nllb_other_til|mtdata|Bible*,lang=udm,docs=14000,per_doc=2' \
  --source 'hf:repo=cis-lmu/Glot500,config=tyv_Cyrl,exclude=dataset=Leipzig*|nllb_other_til|mtdata|Bible*,lang=tyv,docs=15000,per_doc=2' \
  --source hf:repo=ailabykt/sakha-corpus-mono,lang=sah,docs=8000,per_doc=3 \
  --source hf:repo=wikimedia/wikipedia,config=20231101.tt,lang=tat,docs=2000,per_doc=4 \
  --source hf:repo=AigizK/tatar-russian-parallel-corpora,column=tat,lang=tat,docs=999999,per_doc=2,streaming=False \
  --source hf:repo=HuggingFaceFW/fineweb-2,config=tat_Latn,lang=tat,docs=10000,per_doc=3 \
  --source hf:repo=AigizK/bashkir-russian-parallel-corpora,column=ba,lang=bak,docs=30000,per_doc=2 \
  --source hf:repo=Agisight/tyv-rus-200k,column=tyv,lang=tyv,docs=30000,per_doc=2 \
  --source hf:repo=ai-forever/udmurt-corpora,lang=udm,docs=30000,per_doc=2 \
  --source hf:repo=udmurtNLP/zerpal,column=string,lang=udm,docs=20000,per_doc=3 \
  --source hf:repo=d0rj/ru-mhr-parallel,column=mhr,lang=mhr,docs=30000,per_doc=2 \
  --source opus:corpus=translatewiki,src=ce,tgt=en,lang=che \
  --source hf:repo=NM-development/nmd-ce-ru-171k-v0,column=ce,lang=che,docs=999999,per_doc=1,streaming=False \
  --source csv:path=rusentitweet_train.csv,column=text,lang=rus \
  --source jsonl:path=cache/hard_negatives.jsonl \
  --source ukr_tweets:limit=50000 \
  --source mn_social \
  --source kazsandra \
  --source hf:repo=alexantonov/chuvash_mono,column=chv,lang=chv,raw=True,docs=500000,max_chars=512 \
  --source hf:repo=alexantonov/chuvash_russian_parallel,column=chv,lang=chv,raw=True,docs=300000,max_chars=512 \
  --source 'hf:repo=tahrirchi/uz-crawl,files=data/telegram_blogs*,lang=uzn,raw=True,cyr=0.6,docs=400000,max_chars=512' \
  --source hf:repo=averoo/sakha-oscar,lang=sah,raw=True,docs=0,streaming=False,max_chars=512 \
  --source hf:repo=allenai/MADLAD-400,files=data/tyv/tyv_clean_0000.jsonl.gz,lang=tyv,raw=True,docs=0,streaming=False,max_chars=512 \
  --source hf:repo=allenai/MADLAD-400,files=data/sah/sah_clean_0000.jsonl.gz,lang=sah,raw=True,docs=0,streaming=False,max_chars=512 \
  --source hf:repo=YShynkarov/COSMUS,column=document_content,where=language_manual=ukrainian,lang=ukr,raw=True,docs=0,streaming=False,max_chars=512 \
  --source hf:repo=maaxap/BelarusianGLUE,config=besls,column=sentence,lang=bel,raw=True,docs=0,streaming=False \
  --source hf:repo=DGurgurov/macedonian_sa,column=text,lang=mkd,raw=True,docs=0,streaming=False,max_chars=512 \
  --source hf:repo=DGurgurov/bulgarian_sa,column=text,lang=bul,raw=True,docs=0,streaming=False,max_chars=512 \
  --source hf:repo=cardiffnlp/tweet_sentiment_multilingual,files=data/english/train.jsonl,lang=eng,raw=True,docs=0,streaming=False \
  --source hf:repo=cardiffnlp/tweet_sentiment_multilingual,files=data/french/train.jsonl,lang=fra,raw=True,docs=0,streaming=False \
  --source hf:repo=cardiffnlp/tweet_sentiment_multilingual,files=data/german/train.jsonl,lang=deu,raw=True,docs=0,streaming=False \
  --source hf:repo=cardiffnlp/tweet_sentiment_multilingual,files=data/portuguese/train.jsonl,lang=por,raw=True,docs=0,streaming=False \
  --source hf:repo=cardiffnlp/tweet_sentiment_multilingual,files=data/spanish/train.jsonl,lang=spa,raw=True,docs=0,streaming=False \
  --source hf:repo=contemmcm/sentiment140,column=text,lang=eng,raw=True,split=complete,docs=400000,max_chars=512 \
  --source hf:repo=FrancophonIA/french_tweets,lang=fra,raw=True,docs=0,streaming=False,max_chars=512 \
  --source hf:repo=fpaulino/portuguese-tweets,column=tweet_text,lang=por,raw=True,docs=120000,max_chars=512 \
  --source hf:repo=mteb/told-br,lang=por,raw=True,drop_cjk=True,docs=0,streaming=False,max_chars=512 \
  --source hf:repo=NLP-UniBW/tweets_about_german_politicians_jan_feb_2025,where=language=de,lang=deu,raw=True,docs=0,max_chars=512 \
  --source 'hf:repo=pysentimiento/spanish-tweets,files=data/train-00000-of-00166-*.parquet,lang=spa,raw=True,docs=400000,max_chars=512' \
  --source ukr_tweets:lang=ukr,twitter_lang=uk,min_chars=3,max_chars=19,cyr=0.2,limit=200000 \
  --source ukr_tweets:lang=rus,twitter_lang=ru,min_chars=3,max_chars=19,cyr=0.2,limit=100000 \
  --source kazsandra:min_chars=3,max_chars=19,cyr=0.2 \
  --source mn_social:min_chars=3,max_chars=19 \
  --source hf:repo=cardiffnlp/tweet_sentiment_multilingual,files=data/english/train.jsonl,lang=eng,raw=True,docs=0,streaming=False,min_chars=3,max_chars=19 \
  --source hf:repo=cardiffnlp/tweet_sentiment_multilingual,files=data/french/train.jsonl,lang=fra,raw=True,docs=0,streaming=False,min_chars=3,max_chars=19 \
  --source hf:repo=cardiffnlp/tweet_sentiment_multilingual,files=data/portuguese/train.jsonl,lang=por,raw=True,docs=0,streaming=False,min_chars=3,max_chars=19 \
  --source hf:repo=cardiffnlp/tweet_sentiment_multilingual,files=data/german/train.jsonl,lang=deu,raw=True,docs=0,streaming=False,min_chars=3,max_chars=19 \
  --source hf:repo=cardiffnlp/tweet_sentiment_multilingual,files=data/spanish/train.jsonl,lang=spa,raw=True,docs=0,streaming=False,min_chars=3,max_chars=19 \
  --source hf:repo=mteb/told-br,lang=por,raw=True,drop_cjk=True,docs=0,streaming=False,min_chars=3,max_chars=19 \
  --source hf:repo=FrancophonIA/french_tweets,lang=fra,raw=True,docs=0,streaming=False,min_chars=3,max_chars=19 \
  --source hf:repo=contemmcm/sentiment140,column=text,lang=eng,raw=True,split=complete,docs=400000,min_chars=3,max_chars=19 \
  --source hf:repo=DGurgurov/macedonian_sa,column=text,lang=mkd,raw=True,docs=0,streaming=False,min_chars=3,max_chars=19 \
  --source hf:repo=DGurgurov/bulgarian_sa,column=text,lang=bul,raw=True,docs=0,streaming=False,min_chars=3,max_chars=19 \
  --source diverse:lang=rus,pool_repo=HuggingFaceFW/fineweb-2,pool_config=rus_Cyrl,pool_docs=300000,budget=20000 \
  --source diverse:lang=ukr,pool_repo=HuggingFaceFW/fineweb-2,pool_config=ukr_Cyrl,pool_docs=300000,budget=20000 \
  --source hf:repo=YShynkarov/COSMUS,column=document_content,where=language_manual=russian,lang=rus,raw=True,docs=0,streaming=False,max_chars=512 \
  --source csv:path=rusentitweet_train.csv,column=text,lang=rus,min_chars=3,max_chars=19 \
  --source jsonl:path=cache/relabel_rus.jsonl \
  --source diverse:lang=rus,pool_file=cache/wikisource-5da9d07b30.jsonl,budget=6000 \
  --source diverse:lang=kaz,pool_file=cache/kazsandra-eb6571e000.jsonl,budget=16000 \
  --source 'diverse:lang=kir,pool_repo=HPLT/HPLT2.0_cleaned,pool_config=kir_Cyrl,pool_docs=200000,budget=16000,pool_no_chars=ҕһ' \
  --source diverse:lang=tgk,pool_file=cache/hf-b407fce95d.jsonl,budget=16000 \
  --source diverse:lang=uzn,pool_file=cache/hf-5711c74502.jsonl,budget=16000 \
  --source diverse:lang=tat,pool_file=cache/hf-eb368f46e1.jsonl,budget=16000 \
  --source diverse:lang=bak,pool_file=cache/hf-05c948e9d7.jsonl,budget=16000 \
  --source diverse:lang=chv,pool_file=cache/hf-4bb695f530.jsonl,budget=16000 \
  --source diverse:lang=sah,pool_file=cache/hf-fc628b434a.jsonl,budget=16000 \
  --source diverse:lang=tyv,pool_file=cache/hf-b3c61b0e5a.jsonl,budget=16000 \
  --source hf:repo=allenai/MADLAD-400,files=data/ce/ce_clean_0000.jsonl.gz,lang=che,raw=True,docs=0,streaming=False,max_chars=512,cyr=0.6 \
  --source hf:repo=allenai/MADLAD-400,files=data/ce/ce_noisy_0000.jsonl.gz,lang=che,raw=True,docs=0,streaming=False,max_chars=512,cyr=0.6 \
  --source hf:repo=wikimedia/wikipedia,config=20231101.ce,lang=che,docs=30000,per_doc=3 \
  --source 'hf:repo=HPLT/HPLT2.0_cleaned,config=kir_Cyrl,lang=kir,docs=30000,per_doc=4,no_chars=ҕһ' \
  --source hf:repo=allenai/MADLAD-400,files=data/ky/ky_clean_0000.jsonl.gz,lang=kir,raw=True,docs=0,streaming=False,max_chars=512,cyr=0.6 \
  --source hf:repo=wikimedia/wikipedia,config=20231101.ky,lang=kir,docs=10000,per_doc=4 \
  --source 'hf:repo=HPLT/HPLT2.0_cleaned,config=tgk_Cyrl,lang=tgk,docs=30000,per_doc=4,no_chars=ўы' \
  --source hf:repo=alifbank/Tajik,files=tg_sentences.txt,lang=tgk,raw=True,docs=0,streaming=False,min_chars=20,max_chars=512 \
  --source 'hf:repo=allenai/MADLAD-400,files=data/tg/tg_clean_0000.jsonl.gz,lang=tgk,raw=True,docs=0,streaming=False,max_chars=512,cyr=0.6,no_chars=ўы' \
  --source hf:repo=wikimedia/wikipedia,config=20231101.tg,lang=tgk,docs=10000,per_doc=4 \
  --source 'fineweb2:docs_per_lang=3600,per_doc=4,langs=tgk,no_chars=ўы' \
  --source 'hf:repo=tahrirchi/uz-books-v2,files=data/cyr-*.parquet,lang=uzn,docs=30000,per_doc=4' \
  --source hf:repo=cis-lmu/Glot500,config=uzb_Cyrl,lang=uzn,raw=True,docs=200000,max_chars=512,cyr=0.6,where=dataset=Earthlings \
  --cap-per-lang 32000 \
  --short-floor 0.40 \
  --wild-augment 0.3 \
  --short-augment 0.2 \
  --jobs 3 \

