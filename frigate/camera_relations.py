import networkx as nx
from typing import List, Tuple

from frigate.config import CameraRelationsConfig


class CameraRelations:
    def __init__(self, config: CameraRelationsConfig):
        self.config = config
        self.graph = self._create_graph()

    def _create_graph(self):
        # create an empty directed graph
        graph = nx.DiGraph()

        # add nodes for each camera
        for camera_name in self.config.camera_relations:
            graph.add_node(camera_name)

        # add directed edges for each camera relation
        for camera_name, relations in self.config.camera_relations.items():
            for relation in relations:
                # get the name of the other camera in the relation
                other_camera_name = relation.camera_name

                # add a directed edge from the current camera to the other camera,
                # with the relation attributes as edge data
                graph.add_edge(
                    camera_name,
                    other_camera_name,
                    **relation.relation_attributes.dict()
                )

        return graph

    def get_camera_names(self) -> List[str]:
        return list(self.graph.nodes)

    def get_relations(self, camera_name: str) -> List[Tuple[str, dict]]:
        edges = self.graph.edges(camera_name, data=True)
        return [(other_camera, data) for (camera, other_camera, data) in edges]

    def get_adjacent_cameras(self, camera_name: str) -> List[str]:
        successors = list(self.graph.successors(camera_name))
        predecessors = list(self.graph.predecessors(camera_name))
        adjacent_cameras = successors + predecessors
        return adjacent_cameras

    def get_successors(self, camera_name: str) -> List[str]:
        successors = list(self.graph.successors(camera_name))
        return successors

    def get_predecessors(self, camera_name: str) -> List[str]:
        predecessors = list(self.graph.predecessors(camera_name))
        return predecessors

    def get_shortest_path(self, source_camera: str, target_camera: str) -> List[str]:
        try:
            shortest_path = nx.shortest_path(self.graph, source_camera, target_camera)
        except nx.NetworkXNoPath:
            shortest_path = []
        return shortest_path

    def get_distance(self, source_camera: str, target_camera: str) -> float:
        try:
            distance = nx.shortest_path_length(self.graph, source_camera, target_camera)
        except nx.NetworkXNoPath:
            distance = float("inf")
        return distance

    def get_all_relation_attributes(
        self, camera_name_1: str, camera_name_2: str
    ) -> List[Dict[str, Any]]:
        attributes_list = []
        edges = self.graph.edges([camera_name_1, camera_name_2], data=True)
        for (source, target, data) in edges:
            if source == camera_name_1 and target == camera_name_2:
                attributes_list.append(data)
        return attributes_list
