from sqapi.api import SQAPI

api_key = "94dc902ddd52b9e31f9326085116ecf455792216473786295a843ec1" #
sq_connection = SQAPI(api_key=api_key)
print(f"sqapi using login {sq_connection.current_user['username']}")

deployments = ["SS15_FlindersIsland_HarleyPoint_grids",
               "flindersIslandHarleyPoint_08_grids",
               "flindersIslandHarleyPoint_15_grids"]
ann_threshold = 1000

for deployment in deployments:
    results = sq_connection.get("/api/annotation_set",
                        filters=[
                            dict(name="media_collection", op="has", val=
                                 dict(name="media", op="any", val=
                                      dict(name="deployment", op="has", val=
                                            dict(name="name", op="ilike", val=deployment)))),
                            #dict(name="is_qaqc", op="eq", val=True),
                            #dict(name="is_final", op="eq", val=True),
                            #dict(name="is_real_science", op="eq", val=True),
                        ]).execute().json()
    print(f"Deployment {deployment} Annotation sets with annotations > {ann_threshold}")
    for a in results['objects']:
        if a['annotation_count'] >= ann_threshold:
            print(f"annotation_set_id {a['id']} - {a['name']}, {a['annotation_count']} annotations")
