# DATASETS

## LJSpeech

```bash
mkdir -p data && curl -L https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 | tar -xjf - -C data
mv LJSpeech-1.1 LJSpeech
```

## VCTK

```bash
mkdir -p data/VCTK && curl -L https://datashare.ed.ac.uk/download/DS_10283_3443.zip -o data/DS_10283_3443.zip && unzip -q data/DS_10283_3443.zip -d data && ([ -f data/README.txt ] && mv data/README.txt data/VCTK/README_datashare.txt || true) && unzip -q data/VCTK-Corpus-0.92.zip -d data/VCTK && rm data/DS_10283_3443.zip data/VCTK-Corpus-0.92.zip
```

## L2-ARCTIC

1. Fill the request form: https://psi.engr.tamu.edu/l2-arctic-corpus/ (Download section).
2. In the email, choose **“L2-ARCTIC-V5.0 (everything packed)”** to get the full corpus (all 24 speakers).  
   Per-speaker links are for subsets; V1–V4 are legacy; the “Suitcase corpus” is a separate add-on not required.
3. Extract into `./data/L2-ARCTIC/`:
```bash
mkdir -p data/L2-ARCTIC && unzip -q l2arctic_release_v5.0.zip -d data/L2-ARCTIC
cd data/L2-ARCTIC && for z in *.zip; do unzip -q "$z"; done && rm -f *.zip
rm l2arctic_release_v5.0.zip
```
