from collections import Counter

from sqapi.request import query_filter as qf
from sqapi.request import Request
from sqapi.annotate import Annotator
from sqapi.media import SQMediaObject

class Delete(Request):
    def __init__(self, endpoint, data=None, json_data=None, headers=None, **kwargs):
        super().__init__(endpoint, "DELETE", data=data, json_data=json_data, headers=headers or {"Accept": "application/json"}, **kwargs)


class SquidleConnection:
    def __init__(self, sqapi=None):
        self.sqapi = sqapi

    def delete(self, endpoint, data=None, json_data=None, **kwargs) -> Delete:
        return Delete(endpoint, sqapi=self.sqapi, data=data, json_data=json_data, **kwargs)

    def recursive_get(self, endpoint, filter_list, results_per_page=1000):
        """
        Recursive get to retrieve objects
        :param endpoint:
        :param filter_list: list of name,op,val dictionaries
        :param results_per_page:
        :return: list of objects retrieved
        """
        objs = []
        page = 1
        final_page = 10000

        while page < final_page:
            r = self.sqapi.get(endpoint, page=page, results_per_page=results_per_page)
            for f in filter_list:
                r.filter(name=f["name"], op=f["op"], val=f.get("val", None))
            response = r.execute().json()
            if page == 1:
                final_page = response['total_pages']
            print(f"Retrieving page {page} of {final_page}")
            objs += response['objects']
            page += 1
        return objs

    def get_annotation_sets(self, annotation_set_id=None, user_group_id=None, filter_processed_annotation_sets=False,
                            after_date=None):
        """
        Get annotation_set data
        :param annotation_set_id:
        :param user_group_id:
        :param filter_processed_annotation_sets:
        :param after_date:
        :return: dictionary from json response
        """
        r = self.sqapi.get("/api/annotation_set")

        # Filter annotation sets based on ID
        if annotation_set_id:
            r.filter("id", "eq", annotation_set_id)

        # Constrain date ranges to annotation_sets created after a specific date
        if after_date:
            r.filter("created_at", "gt", after_date)

        # Filter annotation_sets based on a user group
        if user_group_id:
            r.filter(name="usergroups", op="any", val=dict(name="id", op="eq", val=user_group_id))

        # Only return annotation_sets that do not already have suggestions from this user
        if filter_processed_annotation_sets:
            r.filter_not(qf("children", "any", val=qf("user_id", "eq", self.sqapi.current_user.get("id"))))
        return r.execute().json()

    def get_media_obj_for_media_ids(self, media_ids, results_per_page=100):
        media_objs = []
        start_idx = 0
        while start_idx < len(media_ids):
            r = self.sqapi.get("/api/media", page=1, results_per_page=results_per_page)
            r.filter(name="id", op="in", val=media_ids[start_idx: min(start_idx + results_per_page, len(media_ids))])
            response = r.execute().json()
            media_objs += response['objects']
            start_idx += results_per_page
        return media_objs

    def get_media_obj_for_media_collection_id(self, media_collection_id, results_per_page=500):
        """
        Recursive select query to get media_objs for media_collection_id
        :param media_collection_id:
        :param results_per_page:
        :return: list of media_objects
        """

        media_objs = []
        page = 1

        response = self.sqapi.get("/api/media", page=page, results_per_page=results_per_page).filter(
            name="media_collections", op="any", val=dict(name="id", op="eq", val=media_collection_id)
        ).order_by(field="timestamp_start", direction="asc").execute().json()
        media_objs += response['objects']
        total_pages = response['total_pages']
        while page < total_pages:
            page += 1
            response = self.sqapi.get("/api/media", page=page, results_per_page=results_per_page).filter(
                name="media_collections", op="any", val=dict(name="id", op="eq", val=media_collection_id)
            ).order_by(field="timestamp_start", direction="asc").execute().json()
            media_objs += response['objects']
        return media_objs

    def get_media_ids_for_annotation_set_ids(self, annotation_set_ids):
        media_ids = []
        for id in annotation_set_ids:
            result = self.sqapi.get(f"/api/annotation_set/{id}")
            media_collection_id = result.execute().json()
            media_collection_id = media_collection_id['media_collection']['id']
            results = self.get_media_ids_for_media_collection_id(media_collection_id)
            media_ids += results
        return media_ids

    def get_annotations_from_set(self, annotation_set_ids=[], label_ids=None, include_annotation_sets=True,
                                 exclude_annotation_sets=[], page=1, needs_review=False,
                                 results_per_page=200):
        """
        Select query on annotation table that returns annotations in or not in the annotation_set_ids list
        depending on include_annotation_sets flag.  Can also select annoations based on review flag and also
        exclude some annotation sets.
        :param annotation_set_ids: list of annotation_set_ids
        :param label_ids: list of label_ids for annotations to be included
        :param include_annotation_sets: Boolean whether to include annotation_set_ids or exclude them
        :param exclude_annotation_sets: list of annotation_set_ids to exclude
        :param page: integer
        :param needs_review: boolean for needs_review flag on annotation
        :param results_per_page: integer
        :return: dictionary response from json query
        """
        request = self.sqapi.get("/api/annotation",
                                 page=page, results_per_page=results_per_page)
        if label_ids is not None:
            request.filter(name="label", op="has", val=dict(name="id", op="in", val=list(label_ids)))
        if include_annotation_sets:
            request.filter(name="annotation_set_id", op="in", val=list(annotation_set_ids))
        else:
            excluded_sets = annotation_set_ids + exclude_annotation_sets
            if len(excluded_sets) > 0:
                request.filter(name="annotation_set_id", op="not_in", val=excluded_sets)
        request.filter(name="needs_review", op="eq", val=str(needs_review))
        request.filter(name="annotation_set", op="has",
                       val=dict(name="is_child", op="eq", val=False))  # Only annotations from parent annotation set
        request.filter(name="point", op="has", val=dict(name="has_xy", op="eq", val=True))
        return request.execute().json()

    def get_all_annotations_from_set(self, annotation_set_ids=None, label_ids=None, include_annotation_sets=True,
                                     exclude_annotation_sets=[], needs_review=False,
                                     results_per_page=400):
        """
        Recursive select query of annotatations in or not in annoation_sets_id depending on include_annotation_sets flag.
        Can exclude annoation sets and use needs_review flag.
        :param annotation_set_ids:
        :param label_ids:
        :param include_annotation_sets:
        :param exclude_annotation_sets:
        :param needs_review:
        :param results_per_page:
        :return: list of annotation dictionaries
        """
        results = []
        page = 1
        ann_results = self.get_annotations_from_set(annotation_set_ids=annotation_set_ids, label_ids=label_ids,
                                                    include_annotation_sets=include_annotation_sets,
                                                    exclude_annotation_sets=exclude_annotation_sets,
                                                    needs_review=needs_review,
                                                    page=page, results_per_page=results_per_page)
        results += ann_results['objects']
        total_pages = ann_results['total_pages']
        print(f"Extracting {total_pages} pages with {ann_results['num_results']} records.")

        while page < total_pages:
            page += 1
            ann_results = self.get_annotations_from_set(annotation_set_ids=annotation_set_ids, label_ids=label_ids,
                                                        include_annotation_sets=include_annotation_sets,
                                                        exclude_annotation_sets=exclude_annotation_sets,
                                                        needs_review=needs_review,
                                                        page=page, results_per_page=results_per_page)
            results += ann_results['objects']

        annotation_set_count = Counter([ann['annotation_set_id'] for ann in results])
        annotation_sets = annotation_set_count.keys()
        to_remove = []
        for a_id in annotation_sets:
            parent_id = self.sqapi.get(f"/api/annotation_set/{a_id}").execute().json()['parent_id']
            if parent_id is not None:
                to_remove.append(a_id)
        print(annotation_set_count)
        print(to_remove)
        print(Counter([ann['label']['id'] for ann in results]))
        return results

    def get_media_ids_for_media_collection_id(self, media_collection_id,
                                              results_per_page=1000):
        """
        Recursive select to get a list of media_ids for a media_collection_id
        :param media_collection_id:
        :param results_per_page:
        :return:
        """
        results = []
        page = 1
        request = self.sqapi.get("/api/media_collection_media",
                                 page=page, results_per_page=results_per_page).filter(name="media_collection_id",
                                                                                      op="==", val=media_collection_id)
        media_ids = request.execute().json()

        results += [m['media_id'] for m in media_ids['objects']]
        total_pages = media_ids['total_pages']

        while page < total_pages:
            page += 1
            request = self.sqapi.get("/api/media_collection_media",
                                     page=page, results_per_page=results_per_page).filter(name="media_collection_id",
                                                                                          op="==",
                                                                                          val=media_collection_id)
            media_ids = request.execute().json()
            results += [m['media_id'] for m in media_ids['objects']]

        return results


class SquidleAnnotator(Annotator):
    """
    Class to do any creating or deleting in squidle. Leverages methods in parent class.
    """

    def __init__(self, **kw):
        super().__init__(**kw)

    def create_media_collection(self, name, description):
        """
        Create a new media_collection
        :param name:
        :param description:
        :return: media_collection_id integer
        """
        data = dict()
        data["user_id"] = self.sqapi.current_user.get("id")
        data['description'] = description or "New media collection."
        data['name'] = name or self.annotator_info

        result = self.sqapi.post("/api/media_collection", json_data=data).execute().json()
        return result['id']

    def create_annotation_set(self, name, description, media_collection_id, label_scheme_id=7, group_id=None):
        """
        :param name: str annotation set name
        :param description: str
        :param media_collection_id: integer
        :param label_scheme_id: integer
        :return: integer annotation_set_id
        """
        data = dict()
        data["user_id"] = self.sqapi.current_user.get("id")
        data['description'] = description or f"New annotation set for media collection {media_collection_id}."
        data['name'] = name or self.annotator_info
        data['label_scheme_id'] = label_scheme_id
        data['media_collection_id'] = media_collection_id

        result = self.sqapi.post("/api/annotation_set", json_data=data).execute().json()
        if group_id is not None:
            group_result = self.sqapi.post(f"/api/annotation_set/{result['id']}/group/{group_id}").execute().json()
        return result['id']

    def add_media_to_collection(self, media_collection_id, media_id_list):
        # todo: make recursive or just select media_id
        request = self.sqapi.get("/api/media",
                                         page=1, results_per_page=1000)
        request.filter(name="media_collection_media", op="any", val={'name': "media_collection_id", 'op': 'eq', 'val': media_collection_id})
        result = request.execute().json()
        media_ids = [m['id'] for m in result['objects']]
        count = 0
        for media_id in media_id_list:
            if media_id not in media_ids:
                count += 1
                self.sqapi.post(f"/api/media_collection/{media_collection_id}/media/{media_id}").execute().json()
        return f"Added {count} media items to media_collection {media_collection_id}"

    def add_annotations_to_annotation_set(self, annotation_set_id, media_collection_id, annotation_list,
                                          label_scheme_id=7):
        # Get media into dictionary by media_id.  Needed for media_obj
        # todo: make this recursive
        if len(annotation_list) > 1000:
            print("fjfjfjfjfjfjfj THIS IS TOO MANY")
            return
        request = self.sqapi.get("/api/media",
                                 page=1, results_per_page=1000)
        request.filter(name="media_collection_media", op="any",
                       val={'name': "media_collection_id", 'op': 'eq', 'val': media_collection_id})
        media_data = request.execute().json()['objects']
        media_lookup = {m['id']: m for m in media_data}

        # Create a simple code lookup based on annotation labels.
        # todo: make this robust to different label schemes
        label_set = list(
            set([a['label']['id'] for a in annotation_list]))
        self.code2label = {l: {'id': l} for l in label_set}

        for annotation in annotation_list:
            # Get the point and media_obj from the annotation
            point = annotation['point']
            m = media_lookup[point['media_id']]
            media_url = m.get('path_best')
            media_type = m.get("media_type", {}).get("name")
            mediaobj = SQMediaObject(media_url, media_type=media_type, media_id=m.get('id'))
            if not mediaobj.is_processed:
                orig_image = mediaobj.data()
            width = mediaobj.width
            height = mediaobj.height
            x = int(point.get('x') * width)
            y = int(point.get('y') * height)
            likelihood = annotation.get('likelihood', 1.0)
            likelihood = 1.0 if likelihood is None else likelihood

            # Create and post point dictionary with annotation_set, media and label data.
            p = self.create_annotation_label_point_px(annotation['label']['id'],
                                                              likelihood=likelihood, comment="Cloned",
                                                              row=x, col=y, width=width, height=height, polygon=None,
                                                              t=point['t'])
            p['annotation_set_id'] = annotation_set_id
            p['media_id'] = mediaobj.id
            if isinstance(p.get('annotation_label'), dict):
                p['annotation_label']['annotation_set_id'] = annotation_set_id
            self.sqapi.post("/api/point", json_data=p).execute()

        return None
