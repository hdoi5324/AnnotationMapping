import requests
import io
from io import BytesIO
import csv
import os
import pandas as pd
import json
from sqapi.api import SQAPI


def recursive_get(sq_connection, endpoint, filter_list, results_per_page=1000):
    """
    Recursive get to retrieve objects
    :sq_connection: instance of SQAPI
    :param endpoint:
    :param filter_list: list of name,op,val dictionaries
    :param results_per_page:
    :return: list of objects retrieved
    """
    objs = []
    page = 1
    final_page = 10000

    while page <= final_page:
        r = sq_connection.get(endpoint, page=page, results_per_page=results_per_page)
        for f in filter_list:
            r.filter(name=f["name"], op=f["op"], val=f.get("val", None))
        response = r.execute().json()
        if page == 1:
            final_page = response['total_pages']
        print(f"Retrieving page {page} of {final_page}")
        objs += response['objects']
        page += 1
    return objs

class create_csv_list():
    
    '''
    Loads API Token from .txt file
    '''
    def load_token(self, HERE):
        with open(HERE + '/API_TOKEN.txt', "r") as file:
            API_TOKEN = file.read().rstrip()
        #print(API_TOKEN)
        return API_TOKEN

    '''
    Get Dataset from SQ which contains all accessible annotation sets
    '''
    def get_annotation_set_ids(self, sq_connection, filters=[], results_per_page=1000):
        results = recursive_get(sq_connection, "/api/annotation_set",
                            filter_list=filters,
                            results_per_page=results_per_page)
        id_list = [r['id'] for r in results]
        return id_list

    '''
    Create a .csv file with all entries
    '''
    def get_annotations_from_annotation_set_ids(self, sq_connection,  id_list, HERE, filters=[]):
        # Create .csv file to append to
        # Export annotations by annotation set id and store in csv
        # Create a csv file with the names of the csvs created.
        NAME = '/Annotation_Sets/Full_Annotation_List.csv'
        with open(HERE + NAME, 'w') as creating_new_csv_file: 
            pass
        include_columns = ["label.id","label.uuid","label.name","tag_names","point.id","point.x","point.y","point.t","point.data","point.media.id","point.media.path_best","point.pose.timestamp","point.pose.lat","point.pose.lon","point.pose.alt","point.pose.dep","label.translated.id","label.translated.uuid","label.translated.name","label.translated.lineage_names","label.translated.translation_info"]
        counter = 0
        fileops = [{"module": "pandas", "method": "json_normalize"},
                                  {"method": "sort_index", "kwargs": {"axis": 1}}]
        # Iterate through all annotation ids
        for i, id in enumerate(id_list):
            #annotation_url = '/' + str(id) + '/export?template=dataframe.csv&disposition=attachment&include_columns=["label.id","label.uuid","label.name","tag_names","point.id","point.x","point.y","point.t","point.data","point.media.id","point.media.path_best","point.pose.timestamp","point.pose.lat","point.pose.lon","point.pose.alt","point.pose.dep","label.translated.id","label.translated.uuid","label.translated.name","label.translated.lineage_names","label.translated.translation_info"]&f={"operations":[{"module":"pandas","method":"json_normalize"},{"method":"sort_index","kwargs":{"axis":1}}]}&q={"filters":[{"name":"point","op":"has","val":{"name":"has_xy","op":"eq","val":true}},{"name":"label_id","op":"is_not_null"}]}&translate={"vocab_registry_keys":["worms","caab","catami"],"target_label_scheme_id":null}'
            print(f"{i} of {len(id_list)}: Processing annotation set {id}")
            filters.append(dict(name="annotation_set_id", op="eq", val=id))
            request = sq_connection.export(f"api/annotation/export",
                       include_columns=include_columns,
                       filters=filters,
                       fileops=fileops,
                       qsparams={"template": "dataframe.csv",
                                 "disposition": "attachment",
                                 "translate": "{\"vocab_registry_keys\":[\"worms\",\"caab\",\"catami\"],\"target_label_scheme_id\":null}"
                     }
                                          )
            result = request.execute()

            decoded_content = result.content.decode('utf-8')
            df = pd.read_csv(io.StringIO(decoded_content))

            if df.size == 0:
                print(f"Warning: no rows for annotation_set_id: {id}")
                continue
            try:
                df = df[['label.id', 'label.name', 'label.uuid', 'point.id', 'point.media.path_best', 'point.x', 'point.y', 'tag_names']]
            except:
                print("Problem with annotation set {}, will be skipped".format(id))
                continue

            # Skips header if this is not the first object
            header = False
            if id == id_list[0]:
                header = True

            #df.drop("Unnamed: 0", axis=1, inplace=True)
            df.to_csv(path_or_buf=HERE+NAME, mode='a', index=False, header=header)
            counter += 1
            print('CSV file number {}/{} saved for id: {}'.format(counter,len(id_list) , id))
        return NAME

if __name__ == "__main__":

    HERE = os.path.dirname(os.path.abspath(__file__))
    data = create_csv_list()
    API_TOKEN = "94dc902ddd52b9e31f9326085116ecf455792216473786295a843ec1" #data.load_token(HERE)
    sq_connection = SQAPI(api_key=API_TOKEN)
    print(f"sqapi using login {sq_connection.current_user['username']}")

    #todo: remove hardcoding of filter attributes.
    annotation_set_filters = [
        dict(name="is_real_science", op="eq", val=True),
        dict(name="is_qaqc", op="eq", val=True),
        dict(name="is_final", op="eq", val=True),
        dict(name="created_at", op="<=", val="01/01/2023"), # REMOVE THIS IF YOU WANT ALL annotation_sets
    ]

    #todo: write id list out to avoid repetitively make the same call on restart.
    #id_list = data.get_annotation_set_ids(sq_connection, annotation_set_filters)
    id_list = [17550]
    # todo: add more filters here if you just want a specific label or matches to a particular label
    annotation_filters = [
                {"name": "point", "op": "has", "val": {"name": "has_xy", "op": "eq", "val": True}},
                {"name": "label_id", "op": "is_not_null"}
        ]
    print('Retrieved a list of {} annotation sets.'.format(len(id_list)))
    NAME = data.get_annotations_from_annotation_set_ids(sq_connection, id_list, HERE, annotation_filters)
    # Name given for testing purpose
    # data.clean_up_dataset(HERE, NAME)
