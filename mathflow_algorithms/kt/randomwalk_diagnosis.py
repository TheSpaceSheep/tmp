from mathflow_algorithms import Algorithm
from mathflow_algorithms import tmpdbinterface as db
import os, random, json, statistics


class RandomWalkDiagnosis(Algorithm):
    """Estimates the user levels in each KC based on known
    levels in neighbouring KCs in the graph (KCs seen in diagnosis quizz)"""
    RANDOM_WALK_COUNT = 150
    @staticmethod
    def get_data_requirements(method=""):
        return [
            "kc_graph",
            "user_id",
            "kc_scope",
            "answers",
        ]

    def _on_call(self, data):
        """Should have the same behavior as compute_random_walks"""
        kc_graph = data["kc_graph"]
        kc_scope = data["kc_scope"]
        user_id = data["user_id"]
        levels = self._retrieve_levels(user_id, kc_graph.all)

        # discard question_card_ids as we don't use them
        answers = {a["kc"]: a["correct"] for a in data["answers"].values()}
        answers[kc_graph.START_NODE_ID] = None
        answers[kc_graph.END_NODE_ID] = None

        # user kc levels go from 0 to 5
        for kc, answer in answers.items():
            if answer:
                levels[kc] = 0.9  # Arbitrary: to get a discovery as first question
            else:
                levels[kc] = 0

        for kc in kc_scope:
            if kc in answers:
                continue

            possible_labels = []
            for _ in range(self.RANDOM_WALK_COUNT):
                count = 0
                current_kc = kc
                is_already_labeled = False

                # walk until we encounter a KC that was seen in the diagnosis quizz
                current_kc = kc
                while not is_already_labeled and count < 100:
                    children = kc_graph.children[current_kc]
                    parents  = kc_graph.parents[current_kc]
                    neighbors = children + parents

                    current_kc = random.choice(neighbors)
                    is_already_labeled = current_kc in answers
                    count += 1

                if current_kc == kc_graph.START_NODE_ID:
                    possible_label = 0
                elif current_kc == kc_graph.END_NODE_ID:
                    possible_label = 0.9
                else:
                    possible_label = levels[current_kc]

                possible_labels.append(possible_label)

            levels[kc] = statistics.mean(possible_labels)

        return levels

    def _retrieve_levels(self, user_id, all_kcs, mode="local"):
        """Retrieves levels for a given user (initialize levels to 0 if they don't exist)"""
        all_levels = {}
        if mode == "local":
            if not os.path.exists("all_levels.json"):
                with open("all_levels.json", 'w') as level_file:
                    all_levels[user_id] = {kc: 0. for kc in all_kcs}
                    json.dump(all_levels, level_file)

            with open("all_levels.json", 'r') as level_file:
                all_levels = json.load(level_file)

            if user_id not in all_levels:
                all_levels[user_id] = {kc: 0. for kc in all_kcs}

            # if the graph has been modified, add the new kcs in all_levels
            if any((kc not in all_levels[user_id]) for kc in all_kcs):
                with open("all_levels.json", 'w') as level_file:
                    all_levels[user_id].update({kc: 0. for kc in all_kcs if kc not in all_levels})
                    json.dump(all_levels, level_file)

            levels = all_levels[user_id]

        elif mode == "database":
            levels = db.retrieve_levels(user_id)
        else:
            raise ValueError(f"Incorrect mode '{mode}'. Mode should be 'database' or 'local'. ")

        return levels

    def _store_levels(self, user_id, levels, mode="local"):
        if mode == "local":
            if not os.path.exists("all_levels.json"):
                return None
            with open("all_levels.json", 'r') as level_file:
                all_levels = json.load(level_file)

            all_levels[user_id] = levels
            with open("all_levels.json", 'w') as level_file:
                json.dump(all_levels, level_file)
        elif mode == "database":
            db.store_levels(user_id, levels)

