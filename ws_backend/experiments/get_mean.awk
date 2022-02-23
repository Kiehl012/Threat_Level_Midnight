awk 'BEGIN{s=0;}{s=s+$1;}END{print s/NR;}' file
