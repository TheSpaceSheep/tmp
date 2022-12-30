import random
from mathflow_algorithms import Algorithm


class DummyRecommendation(Algorithm):
    """Recommends a random KC"""
    @staticmethod
    def get_data_requirements(method=""):
        return ["kc_scope"] if method=="__call__" else []

    def _on_call(self, data):
        kc_scope = data["kc_scope"]
        return random.choice(kc_scope)
