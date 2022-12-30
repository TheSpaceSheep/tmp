from mathflow_algorithms import Algorithm
import random


class DummyDiagnosis(Algorithm):
    """Gives random levels to any user"""
    @staticmethod
    def get_data_requirements(method=""):
        return ["kc_graph"]

    def _on_call(self, data):
        all_kcs = data["kc_graph"].all
        return {kc: random.random() for kc in all_kcs}
