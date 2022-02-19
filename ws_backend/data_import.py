from sqlalchemy import exc
from sqlalchemy import Column, Integer, String, Float
import requests
from models import Characters, Episodes, Scripts
from database import db_session
import csv


def import_ep_csv(ep_csv, dbh):
    # Original air date,overall,Season,Episode,Line_ID,Title,
    # Directed by,Written by
    with open(ep_csv, newline='\n', encoding='windows-1252') as csvfile:
        ep_info = csv.reader(csvfile, delimiter=',')
        for row in ep_info:
            cur_ep_date = row[0]
            cur_ep_id = row[1]
            cur_sea = row[2]
            cur_episode = row[3]
            cur_line_id = row[4]
            cur_title = row[5]
            cur_director = row[6]
            cur_writer = row[7]
            new_ep = Episodes(episode_id=cur_ep_id,
                              title=cur_title,
                              season_no=cur_sea,
                              episode_no=cur_episode,
                              air_date=cur_ep_date,
                              line_id=cur_line_id,
                              writer=cur_writer,
                              director=cur_director)
            dbh.add(new_ep)
        dbh.commit()


def import_line_csv(lines_csv, dbh):
    with open(lines_csv, newline='\n') as csvfile:
        clean_script = csv.reader(csvfile, delimiter=',')
        # ['41689', 'Michael', 'I got it from NewYorkTimes.com', \
        # '7', '10', '710']
        for row in clean_script:
            id = int(row[0])
            char = row[1]
            line = row[2]
            season_no = int(row[3])/100
            episode_no = int(row[4])
            ep_id = int(row[5])
            sentiment_score = float(row[6])
            new_script = Scripts(line_id=id,
                                 emp_name=char,
                                 season=season_no,
                                 episode=episode_no,
                                 sentiment=sentiment_score,
                                 line=line,
                                 episode_id=ep_id)
            dbh.add(new_script)
        dbh.commit()


def import_data():
    line_filename = 'cleaned_scripts.csv'
    ep_filename = 'Episode_Detail.csv'
    import_line_csv(line_filename, db_session)
    import_ep_csv(ep_filename, db_session)
