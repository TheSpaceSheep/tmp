from mathflow_algorithms import Algorithm
from mathflow_algorithms import tmpdbinterface as db
import os, random, json


class KCBandit(Algorithm):
    def __init__(self):
        self._epsilon = 0.1


    @staticmethod
    def get_data_requirements(method=""):
        call_keys = [
            "kc_graph",
            "kc_scope",
            "user_id",
            "user_levels",
            "activity_counts"
        ]
        update_keys = [
            "kc_graph",
            "session_kc_id",
            "session_history",
            "user_id",
        ]
        if method == "__call__":
            return call_keys
        elif method == "update":
            return update_keys
        else:
            return call_keys + update_keys


    def _on_call(self, data):
        """
        Should have the same behavior as the function choose_kc_from_query
        """
        all_kcs         = data["kc_graph"].all
        kc_parents_ids  = data["kc_graph"].parents
        epsilon         = self._epsilon
        omegas          = self._retrieve_omegas(data["user_id"], all_kcs)

        kc_without_parents = [kc for kc in kc_parents_ids if len(kc_parents_ids[kc])==0]
        zpd = self._get_zpd(kc_without_parents, data)

        if len(zpd) == 0:
            return random.choice(all_kcs)

        omegas_list = list(omegas.values())
        sum_omegas = sum(omegas)
        omegas_normalized = [w / sum_omegas for w in omegas_list]
        user_proba_kc_in_zpd = [(w * (1 - epsilon) + epsilon) for w in omegas_normalized]
        user_proba_kc_normalized_in_zpd = [(x / sum([y for y in user_proba_kc_in_zpd])) for x in user_proba_kc_in_zpd]
        chosen_kc_id = random.choices(zpd, weights=user_proba_kc_normalized_in_zpd, k=1)[0]
        return chosen_kc_id


    def update(self, data):
        """
        Should have the same behavior as update_user_kc_omega
        """
        user_id = data["user_id"]
        kc = data["session_kc_id"]
        all_kcs = data["kc_graph"].all
        omegas = self._retrieve_omegas(user_id, all_kcs)
        session_history = data["session_history"]   # list of dict {type: ..., score: ...}

        new_reward = self._compute_reward(session_history)
        omegas[kc] += 0.1 * new_reward

        return omegas
        #self._store_omegas(user_id, omegas)


    def _compute_reward(self, session_history):

        elements_count = 4
        success_list = [1 if (
                   (session["type"] == "SPRINT"         and session["score"] > 0)
                or (session["type"] == "DISCOVER_QUIZZ" and session["score"] > 1)
                or (session["type"] == "CLIMB"          and session["score"] > 2)
        ) else 0 for session in session_history]

        if len(success_list) < elements_count:
            new_reward = 0
        else:
            new_reward = ((sum(success_list[elements_count // 2:elements_count]) / (elements_count // 2)) -
                          (sum(success_list[0:elements_count // 2]) / (elements_count - (elements_count // 2))))

        return new_reward


    def _get_zpd(self, kcs, data, get_parents=True, get_children=True):
        zpd = []
        for kc in kcs:
            user_level = data["user_levels"][kc]
            kc_to_append = None
            activities_done_in_kc_count = data["activity_counts"]
            if activities_done_in_kc_count > 3:
                if get_parents and user_level < 1.5:
                    parents = data["kc_graph"].parents[kc]
                    zpd += self._get_zpd(parents, data, get_children=False)
                elif get_children  and user_level > 4:
                    children = data["kc_graph"].children[kc]
                    zpd += self._get_zpd(children, data, get_parents=False)
                else:
                    kc_to_append = kc
            else:
                kc_to_append = kc

            if kc_to_append is not None:
                zpd.append(kc_to_append)

        return zpd


    def _retrieve_omegas(self, user_id, all_kcs, mode="local"):
        """
        Use "local" mode to store internal parameters in a local file
            "database" mode to store them in the database
        """
        all_omegas = {}
        if mode == "local":
            if not os.path.exists("all_omegas.json"):
                with open("all_omegas.json", 'w') as omega_file:
                    all_omegas[user_id] = {kc: 1. for kc in all_kcs}
                    json.dump(all_omegas, omega_file)

            with open("all_omegas.json", 'r') as omega_file:
                all_omegas = json.load(omega_file)

            if user_id not in all_omegas:
                all_omegas[user_id] = {kc: 1. for kc in all_kcs}

            # if the graph has been modified, add the new kcs in all_omegas
            if any((kc not in all_omegas[user_id]) for kc in all_kcs):
                with open("all_omegas.json", 'w') as omega_file:
                    all_omegas[user_id].update({kc: 1. for kc in all_kcs if kc not in all_omegas})
                    json.dump(all_omegas, omega_file)

            omegas = all_omegas[user_id]

        elif mode == "database":
            omegas = db.retrieve_omegas(user_id)
        else:
            raise ValueError(f"Incorrect mode '{mode}'. Mode should be 'database' or 'local'. ")

        return omegas


    def _store_omegas(self, user_id, omegas, mode="local"):
        """Retrieves omegas for a given user (initialize omegas to 1 if they don't exist)"""
        if mode == "local":
            if not os.path.exists("all_omegas.json"):
                return None
            with open("all_omegas.json", 'r') as omega_file:
                all_omegas = json.load(omega_file)

            all_omegas[user_id] = omegas
            with open("all_omegas.json", 'w') as omega_file:
                json.dump(all_omegas, omega_file)
        elif mode == "database":
            db.store_omegas(user_id, omegas)

