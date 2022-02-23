export POSTGRES_USER=postgres; 
export POSTGRES_PASSWORD=dae10Bksd;
export POSTGRES_DB=script_analysis;
export APP_SECRET_KEY='\xe8V\xdcHAoQ\xe8\x9d\x98N\xbdR8\xaf.\x1a\xe3\x07+\xc4\r\xa0B^\x16\xa6\xeb\xda\x03\xc6\x15';

python models.py 
python data_import.py
