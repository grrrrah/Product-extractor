from identify_nps import get_continuous_chunks, extract_entity
import os
import glob
import pandas as pd
pd.set_option('display.max_rows', None)

files = list(glob.glob(os.path.join('test data','*.*')))


all_entities_identified = []
for file in files:
    en = extract_entity(file)
    all_entities_identified.append(en)


df = pd.DataFrame(all_entities_identified, columns=['Entity label','Product Name'])

print(df)
