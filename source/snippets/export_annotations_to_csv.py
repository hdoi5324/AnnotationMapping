import io
import json

import pandas as pd
from sqapi.api import SQAPI

api_key = "INSERTAPIKEY"
sq_connection = SQAPI(api_key=api_key)
print(f"sqapi using login {sq_connection.current_user['username']}")

annotation_set_id = 17550  # Red cup sponge test
target_label_scheme_id = 75  # translate labels to Seamap Australia (75)
csv_filename = f"./annotations_{annotation_set_id}.csv"

filters = [
    dict(name="point", op="has", val=dict(name="has_xy", op="eq", val=True)),
    dict(name="label_id", op="is_not_null"),
    dict(name="annotation_set_id", op="eq", val=annotation_set_id)
]
include_columns = ["label.id", "label.uuid", "label.name", "tag_names", "point.id", "point.x", "point.y", "point.t",
                   "point.data", "point.media.id", "point.media.path_best", "point.pose.timestamp", "point.pose.lat",
                   "point.pose.lon", "point.pose.alt", "point.pose.dep", "label.translated.id", "label.translated.uuid",
                   "label.translated.name", "label.translated.lineage_names", "label.translated.translation_info"]

fileops = [dict(module="pandas", method="json_normalize"),  # flatten structure
                dict(method="sort_index", kwargs=dict(axis=1))]  # sorting

translate = dict(vocab_registry_keys=["worms", "caab", "catami"],
                 target_label_scheme_id=target_label_scheme_id)

# Build request then convert results to pandas dataframe
request = sq_connection.export(f"api/annotation/export",
                               include_columns=include_columns,
                               filters=filters,
                               fileops=fileops,
                               qsparams=dict(template="dataframe.csv",
                                             disposition="attachment",
                                             translate=json.dumps(translate)
                                             )
                               )
result = request.execute()

decoded_content = result.content.decode('utf-8')
df = pd.read_csv(io.StringIO(decoded_content))

if df.size == 0:
    print(f"Warning: no rows for annotation_set_id: {annotation_set_id}")

df.to_csv(path_or_buf=csv_filename, mode='a', index=False, header=True)
print(f"Saved file to {csv_filename} with {df.shape[0]} rows")
