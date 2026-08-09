# Missingness Explorer

Interactive visual-analytics tool for exploring the structure and mechanisms of
missing data. Demonstrated on the UK Crop Microbiome Cryobank (ENA PRJEB58189).

## Data
curl -o data/raw/PRJEB58189.tsv "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=PRJEB58189&result=read_run&fields=all&format=tsv&limit=0"

## Pipeline
pip install pandas numpy pyarrow openpyxl
python pipeline/missingness.py --input data/raw/PRJEB58189.tsv --outdir web --title "UK Crop Microbiome Cryobank"

## Front end
cd web && python -m http.server 8000    # open http://localhost:8000