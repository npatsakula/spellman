# Wild-register UGC corpus candidates (rusentitweet analogs, other languages)

Research date: 2026-08-16. All URLs verified live by research agents (page fetches,
row previews, HTTP probes; nothing downloaded). Companion to the church/bible lane:
this file covers the "wild social-media register" lane. Labels (sentiment etc.) are
irrelevant — raw text is the point; prefer URLs/@mentions intact (our featurizer
canonicalizes them).

Legend: ⭐ = true rusentitweet analog (raw UGC at scale). "Gated (click)" = HF
login + accept terms, still scriptable.

## Best picks per language

| lang | primary (register anchor) | volume backup | notes |
|---|---|---|---|
| ukr | ⭐ saganoren/ukr-twi-corpus — 1.85M raw tweets, GitHub `corpus.tar.xz` (53MB), no license file | UberText 2.0 social layer (Telegram, CC BY-NC-SA); YShynkarov/COSMUS 12k MIT w/ per-text lang labels (ua/ru/code-switch) | only large Ukrainian tweet dump that exists; pre-2022 usage — pair with Telegram-era data |
| bel | maaxap/BelarusianGLUE ~2-3k genuine UGC rows (apache-2.0) | FineWeb-2 bel_Cyrl 2.1M docs (web); CC100-be as trasianka/RU-contaminated hard negatives | **no wild corpus exists** — Belarusian social media is mostly Russian |
| kaz | ⭐ kurumikz/telegram-corpus-russian-kazakh — 1.49M raw Telegram lines, explicitly uncleaned (CC-BY-NC-SA) | IS2AI KazSAnDRA ~220k reviews (CC-BY-4.0, ungated GitHub) | RU+KZ mixed — we LID-filter; NC license |
| kir | Leipzig kir_community_2017 ~1M sentences (web/comment pages) | averoo/kyrgyz_mono 752k docs (no license, some contamination) | **weakest class**; no tweet/comment corpus exists (confirmed by KyrgyzNLP survey) |
| tgk | Leipzig tgk_community_{2017,2021,2022} ~1M sentences each, ungated curl | muhtasham/tajik-corpus (gated click, CC-BY-4.0, but sanitized — anti-wild) | no social dump exists |
| uzn | ⭐ tahrirchi/uz-crawl `telegram_blogs` — 368k Telegram posts, @mentions intact, apache-2.0 | Leipzig uzb_community_2017 ~1M (2017 vintage → naturally Cyrillic) | **script-filter to Cyrillic** (mixed); all "uzbek sentiment" datasets are Latin bait |
| tat | ⭐ TatarNLPWorld/sovet_kinesh-vk (138k comments) + allahtan-vk (86k comments), apache-2.0, gated click; + vk-groups 57k posts (MIT) | yasalma/community-oscar-tatar (gated, sample) | heavy tat–ru code-switching (realistic); no Tatar tweet corpus exists |
| bak | — (nothing UGC) | BashkirNLPWorld/bashkir-web-corpus 72k docs (CC BY-SA, gated click); slone/bak_rus_3M2023_scored 3.7M pairs (take `ba` side, sim-filter) | VK scraping is the only route to wild Bashkir |
| chv | — (nothing UGC) | ⭐-ish alexantonov/chuvash_mono 2.9M sentences, CC0, **ungated** (raw web mix incl. comment fragments, OCR noise) | + chuvash_russian_parallel 1.46M pairs (CC0, ungated, take chv side) |
| sah | — (nothing UGC) | averoo/sakha-oscar 8.8k (raw crawl); FineWeb-2 sah_Cyrl 73k; ailabykt/sakha-corpus-mono (gated click, CC-BY-4.0, news/OCR — already known) | Sakha-Language-Processing GitHub ships scrapers (kyym.ru, keskil14.ru forum) but no data |
| tyv | — | MADLAD-400 tyv 9.1k docs (ODC-BY); FLORES+ tyv 2k (eval only) | **effectively nothing exists** |
| bul | DGurgurov/bulgarian_sa 7.9k informal movie-site comments (MIT) | CLASSLA-web.bg 2.0 (CC0, no registration) filtered genre=Forum/Opinion | TRACES tweet datasets = ID-only + gated + unrehydratable (dead) |
| mkd | ⭐ mteb/MacedonianTweetSentimentClassification 9.7k raw tweets (CC BY-NC-SA; @ stripped to bare names) | DGurgurov/macedonian_sa 8.2k (MIT, same base corpus — dedup!); CLASSLA-web.mk genre=Forum (CC0) | easiest Balkan pick; "mkoffenseval" does not exist (verified) |
| srp | — (all Serbian tweet datasets are Latin-dominant; Twitter-HBS even transliterates Cyrillic away) | **CLASSLA-web.sr 2.0** (CC0) filter `script=Cyrillic AND genre=Forum` — per-text script metadata, .срб-TLD crawl; SentiComments.SR.orig 4.1k (CC BY-NC-SA, mixed script — filter by ratio) | srp_Cyrl needs the CLASSLA route; reldi_sr/AbCoSER are small/Latin |
| mon | ⭐ ganaxy/diploma `relabeled_v7_corrected.csv` — 10k raw social comments (news.mn/gogo.mn/FB/YouTube/X), `text_raw` column, no license (thesis repo) | 11-11.mn complaints ~80k (Kaggle); CC-100 mn; OSCAR-2201 mn (gated manual, rawest) | beware Inner-Mongolia datasets = traditional script (MC², Mongolian-pretrain-d) — rejected |
| eng | ⭐ contemmcm/sentiment140 — 1.6M raw 2009 tweets, real @handles/URLs | cardiffnlp/tweet_eval (sanitized @user/http) | |
| spa | ⭐ pysentimiento/spanish-tweets — 600M tweets (stream it!), 7-8% pt/en/ca noise (confusion-pair material) | johnatanebonilla/tweet_hisp 217M (thin provenance) | |
| fra | FrancophonIA/french_tweets 21.6k + UMSAB french 3k | — | **gap**: tilomilo 1.5M is 401/dead; no open ≥50k French tweet corpus |
| por | ⭐ fpaulino/portuguese-tweets ~115k raw (EU-PT) + mteb/told-br 21k BR-PT (CC-BY-SA) | UMSAB portuguese | both PT variants covered |
| deu | ⭐ NLP-UniBW/tweets_about_german_politicians_jan_feb_2025 — 829k tweets w/ `language` column (raw replies) | NLP-UniBW tweets_dataset_jan_feb_big_deduplicated 30.7M | political skew (query-collected) |

## Multilingual one-shot

`cardiffnlp/tweet_sentiment_multilingual` (UMSAB) — 3k raw tweets × each of
eng/fra/deu/por/spa (+ara/hin/ita), ungated, one download. Partially
@user/http-anonymized. Fine realism anchor, no scale.

## Access commands (primary picks)

```bash
# ukr
git clone https://github.com/saganoren/ukr-twi-corpus.git   # corpus.tar.xz inside
# kaz / uzn / tat / chv (HF)
hf download kurumikz/telegram-corpus-russian-kazakh --repo-type dataset
hf download tahrirchi/uz-crawl --repo-type dataset --include "data/telegram_blogs*"
hf download TatarNLPWorld/sovet_kinesh-vk --repo-type dataset        # gated click
hf download alexantonov/chuvash_mono --repo-type dataset
# mkd / bul / mon
hf download mteb/MacedonianTweetSentimentClassification --repo-type dataset
hf download DGurgurov/bulgarian_sa --repo-type dataset
curl -L "https://raw.githubusercontent.com/ganaxy/diploma/master/sample%20scores/relabeled_v7_corrected.csv"
# kir / tgk / uzn-cyr (Leipzig)
curl -O https://downloads.wortschatz-leipzig.de/corpora/kir_community_2017.tar.gz
curl -O https://downloads.wortschatz-leipzig.de/corpora/tgk_community_2022.tar.gz
curl -O https://downloads.wortschatz-leipzig.de/corpora/uzb_community_2017.tar.gz
# srp (CC0, big — 50GB annotated; use plain JSONL, filter script=Cyrillic genre=Forum)
curl -LO "https://www.clarin.si/repository/xmlui/handle/11356/2079/CLASSLA-web.sr.2.0.jsonl.gz?sequence=1&isAllowed=y"
# Latin G10
hf download contemmcm/sentiment140 --repo-type dataset
hf download fpaulino/portuguese-tweets --repo-type dataset
hf download NLP-UniBW/tweets_about_german_politicians_jan_feb_2025 --repo-type dataset
hf download pysentimiento/spanish-tweets --repo-type dataset    # 52GB — stream instead
hf download cardiffnlp/tweet_sentiment_multilingual --repo-type dataset
```

## Wired source specs (2026-08-17)

Adapters: `hf` gained raw mode + row gates (`raw,min_chars,max_chars,where,files,cyr,drop_cjk,split`),
`leipzig` gained `cyr`; new `spellman_train/sources/ugc.py` (`ukr_tweets`, `mn_social`, `kazsandra`).
Every spec below is cached under `cache/<name>-<hash>.jsonl` and verified against the
VALIDATION.md counts:

```bash
uv run spellman-train mix --out <mix_dir> \
  --source ukr_tweets:limit=400000 \
  --source mn_social \
  --source kazsandra \
  --source 'hf:repo=alexantonov/chuvash_mono,column=chv,lang=chv,raw=True,docs=500000,max_chars=512' \
  --source 'hf:repo=alexantonov/chuvash_russian_parallel,column=chv,lang=chv,raw=True,docs=300000,max_chars=512' \
  --source 'hf:repo=tahrirchi/uz-crawl,files=data/telegram_blogs*,lang=uzn,raw=True,cyr=0.6,docs=400000,max_chars=512' \
  --source 'leipzig:corpus=kir_community_2017,lang=kir,limit=250000' \
  --source 'leipzig:corpus=tgk_community_2022,lang=tgk,cyr=0.6,limit=500000' \
  --source 'leipzig:corpus=uzb_community_2017,lang=uzn,cyr=0.6,limit=400000' \
  --source 'hf:repo=averoo/sakha-oscar,lang=sah,raw=True,docs=0,streaming=False,max_chars=512' \
  --source 'hf:repo=allenai/MADLAD-400,files=data/tyv/tyv_clean_0000.jsonl.gz,lang=tyv,raw=True,docs=0,streaming=False,max_chars=512' \
  --source 'hf:repo=allenai/MADLAD-400,files=data/sah/sah_clean_0000.jsonl.gz,lang=sah,raw=True,docs=0,streaming=False,max_chars=512' \
  --source 'hf:repo=YShynkarov/COSMUS,column=document_content,where=language_manual=ukrainian,lang=ukr,raw=True,docs=0,streaming=False,max_chars=512' \
  --source 'hf:repo=maaxap/BelarusianGLUE,config=besls,column=sentence,lang=bel,raw=True,docs=0,streaming=False' \
  --source 'hf:repo=DGurgurov/macedonian_sa,column=text,lang=mkd,raw=True,docs=0,streaming=False,max_chars=512' \
  --source 'hf:repo=DGurgurov/bulgarian_sa,column=text,lang=bul,raw=True,docs=0,streaming=False,max_chars=512' \
  --source 'hf:repo=cardiffnlp/tweet_sentiment_multilingual,files=data/english/train.jsonl,lang=eng,raw=True,docs=0,streaming=False' \
  --source 'hf:repo=cardiffnlp/tweet_sentiment_multilingual,files=data/french/train.jsonl,lang=fra,raw=True,docs=0,streaming=False' \
  --source 'hf:repo=cardiffnlp/tweet_sentiment_multilingual,files=data/german/train.jsonl,lang=deu,raw=True,docs=0,streaming=False' \
  --source 'hf:repo=cardiffnlp/tweet_sentiment_multilingual,files=data/portuguese/train.jsonl,lang=por,raw=True,docs=0,streaming=False' \
  --source 'hf:repo=cardiffnlp/tweet_sentiment_multilingual,files=data/spanish/train.jsonl,lang=spa,raw=True,docs=0,streaming=False' \
  --source 'hf:repo=contemmcm/sentiment140,column=text,lang=eng,raw=True,split=complete,docs=400000,max_chars=512' \
  --source 'hf:repo=FrancophonIA/french_tweets,lang=fra,raw=True,docs=0,streaming=False,max_chars=512' \
  --source 'hf:repo=fpaulino/portuguese-tweets,column=tweet_text,lang=por,raw=True,docs=120000,max_chars=512' \
  --source 'hf:repo=mteb/told-br,lang=por,raw=True,drop_cjk=True,docs=0,streaming=False,max_chars=512' \
  --source 'hf:repo=NLP-UniBW/tweets_about_german_politicians_jan_feb_2025,where=language=de,lang=deu,raw=True,docs=0,max_chars=512' \
  --source 'hf:repo=pysentimiento/spanish-tweets,files=data/train-00000-of-00166-*.parquet,lang=spa,raw=True,docs=400000,max_chars=512'
```

Gotchas baked into these specs: umsab is a legacy script repo (read via `files=` +
builder, bypassing the script); sentiment140's single split is `complete`;
kazsandra dedups `custom_id` over the canonical zips only; ukr_tweets keeps only
Twitter-self-labeled `lang=uk` rows with a Cyrillic gate. **`max_chars=512` is
not optional on raw doc-bearing sources** — OSCAR/MADLAD rows reach 190k chars,
and unclamped they blow up downstream vectorized passes (the hygiene judge
materializes `[tokens × classes]` per chunk). DGurgurov macedonian_sa/bulgarian_sa
(MIT) replace the NC-licensed mteb Macedonian tweets of the old recipe. These 26
specs yield ~3.5M wild/UGC rows; the combined v4 recipe (standing 15:15 sources +
these, minus the NC mteb mkd source and the subsumed windowed chuvash_mono) lives
in `data_mix2/manifest.json` after the first run.

Blocked on the HF gate (manual approval — `gated: manual`, still 403 for the
vpermilp token as of 2026-08-17): add this spec once the BashkirNLPWorld owners
approve access:

    --source 'hf:repo=BashkirNLPWorld/bashkir-web-corpus,lang=bak,docs=0,per_doc=4'

(71,567 document-level web docs, CC BY-SA 4.0 — windowed sampling like
FineWeb-2, not raw mode; ungated bak volume meanwhile: `slone/bak_rus_3M2023_scored`
and `AigizK/bashkir-russian-parallel-corpora`, both parquet with a `ba` column.)

## Fetch + validation results (2026-08-17)

All slugs under `train/cache/raw/<slug>/`, each with VALIDATION.md (full stats + samples).
Pending: tat/bul/mkd/srp batch (agent cancelled — TatarNLPWorld vk dumps, DGurgurov ×2,
CLASSLA-web.sr sample) and bashkir-web-corpus (HF gate approval needed).

| slug | rows | text col | status | key finding |
|---|---|---|---|---|
| ukr-twi-corpus | 1,854,993 | text (CSV, embedded newlines — parse with pyarrow `newlines_in_values`) | needs-LID-filter | 78.8% cyr; Twitter-lang 84.3% uk; ~75-85% truly Ukrainian; NO license stated |
| cosmus | 12,224 | document_content (parquet; filename has a space) | gold-ready (labeled) | MIT; `language_manual`: ukr 6,886 / rus 3,130 / surzhyk 1,747 — slice ukrainian for train, keep rest as eval |
| belarusian-glue | 2,000 UGC | sentence (besls config only; arrow) | gold-ready (tiny) | apache-2.0; only besls is UGC; other configs formal/MT — exclude |
| kazsandra | 175,158 unique | text (zips→CSV) | needs-LID-filter | CC-BY-4.0 (HF card only); use ONLY ib/valid/test zips, dedup custom_id (ros/rus zips duplicate!); 80.8% carry Kazakh letters |
| chuvash-mono | 2,917,415 | chv (parquet) | gold-ready | CC0 ungated; 15/15 pure; 98.8% Chuvash letters |
| chuvash-parallel | 1,461,485 | chv (drop ru) | gold-ready | CC0; only 4.9% overlap with mono → ~4.3M combined chv |
| bashkir-web-corpus | 0 | — | **blocked: gated** | "Access denied. This repository requires approval." — user must click approval on HF |
| kir-community-2017 | 251,608 | sentences.txt field 2 | gold-ready | 100% cyr full-scan; 0.04% RU-look; Leipzig .txt layout (no URL column) |
| tgk-community-2022 | 941,793 | sentences.txt field 2 | gold-ready | 97.8% cyr (drop 2.2% Latin → 920,886); 1.1% RU-bleed |
| uzb-community-2017 | 663,119 (320,031 cyr) | sentences.txt field 2 | needs-script-filter | 48/52 mixed script; Cyrillic cut 92.3% ўқғҳ, ~0% RU; Latin half = Latin-uzb material |
| uz-crawl-telegram | 368,017 (277,867 ≥60% cyr) | text (parquet) | needs-script-filter | Apache-2.0; 76% cyr, 97.9% Uzbek letters; 63% rows carry @handles → strip `@\w+` |
| mn-social-comments | 10,000 | text_raw (CSV) | gold-ready | 0/10k traditional script (pure Cyrillic khalkha); 15/15 MN; license unstated (thesis repo) |
| sakha-oscar | 8,783 | text (parquet) | needs-LID-filter | boilerplate dupes (CC footers), dedupe; license unstated (OSCAR deriv) |
| madlad-tyv | 9,083 clean | text (jsonl.gz `data/tyv/tyv_clean_0000.jsonl.gz`) | needs-LID-filter (mild) | ODC-BY; 10.2% no-ө/ү/ң but mostly diacritic-stripped Tuvan, not RU |
| sakha-madlad | 29,169 clean | text (`data/sah/sah_clean_0000.jsonl.gz`) | needs-LID-filter | ODC-BY; 10.8% no-Sakha-grapheme (RU commercial pages); **skip data-v1p5/ (exact duplicate, 1.05GB — deletable)** |
| umsab | 24,264 (3,033×8) | text (jsonl per lang dir) | gold-ready (small) | CC-BY 3.0; @user-anonymized; 15,165 rows across our 5 Latin langs; 1.1% texts have literal \uXXXX escapes — unescape in adapter |
| sentiment140 | 1,600,000 | text (CSV) | gold-ready (eng) | raw handles/URLs; unstated license |
| french-tweets | 21,591 (NOT 215k) | JSON array of strings | gold-ready (fra) | 48% URL-laden truncated tweets |
| portuguese-tweets | 114,998 | tweet_text (parquet) | needs-LID-filter (light) | skews BR-PT despite EU-PT claim; some es rows; uuid-artifact noise rows |
| told-br | 21,000 | text (parquet) | gold-ready after mojibake repair | CC-BY-SA-4.0; 5.24% UTF-8→GBK mojibake rows (`bostalh茫o`) — repair or drop |
| deu-politicians-2025 | 829,191 | text (single 206MB parquet) | needs-LID-filter (trivial) | `language` col: de 87.2% — filter on it; PII columns (username/profile_id/link) — drop at ingestion |
| spanish-tweets-sample | 3,598,995 (1 shard; ~597M total) | text (parquet) | volume-only | ~5% pt/en contamination → LID pass; one shard is plenty |

Immediate gold pool without any filtering: chv 4.4M, tgk 921k, eng 1.6M, kir 252k,
deu 723k (post language-col filter), por ~136k, fra 21.6k, mn 10k, ukr 12.2k
(labeled slice), bel 2k.

## Strategic notes

1. Six true analogs exist (ukr/kaz/uzn/tat/mkd/mon + the Latin G10 picks); for
   bel/kir/tgk/bak/sah/tyv/srp-cyrl nothing public is wild — the only route is
   scraping VK/Telegram/forums ourselves (the region's UGC lives on VK/Telegram,
   not X, post-2022).
2. Script traps: uzn must be Cyrillic-filtered (Latin is default in modern Uzbek
   datasets); srp Twitter is Latin-dominant (CLASSLA-web.sr has per-text script
   metadata); Inner-Mongolian "Mongolian" datasets are traditional script.
3. License watch: kurumikz (CC-BY-NC-SA), mteb mkd (CC BY-NC-SA), SentiComments.SR
   (CC BY-NC-SA), UberText social (CC BY-NC-SA) — non-commercial; sentiment140,
   fpaulino, NLP-UniBW, ganaxy have no license stated.
4. Contamination is a feature where controlled: kurumikz RU+KZ mix, spanish-tweets
   7-8% pt/en, CC100-be trasianka — good hard-negative material after our LID
   bootstrapping filter.
