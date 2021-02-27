from .systems import RSClusterBased1, RSClusterBased2, RSUserBased
from enum import Enum
from json import dumps
from datetime import datetime


class RecommendationType(Enum):
    USER_BASED = "USER_BASED"
    CLUSTER_BASED_1 = "CLUSTER_BASED_1"
    CLUSTER_BASED_2 = "CLUSTER_BASED_2"
    ALL = "ALL"


class Merger:
    def __init__(self):
        self.rs1 = RSUserBased()
        self.rs2 = RSClusterBased1()
        self.rs3 = RSClusterBased2()

    def get_recommendation(self, cli_id: int, rt: RecommendationType, n=0):
        r = None

        if rt == RecommendationType.USER_BASED:
            r = self.rs1.get_recommendation(cli_id)
        elif rt == RecommendationType.CLUSTER_BASED_1:
            r = self.rs2.get_recommendation(cli_id)
        elif rt == RecommendationType.CLUSTER_BASED_2:
            r = self.rs3.get_recommendation(cli_id)
        elif rt == RecommendationType.ALL:
            return self.get_merged_recommendation(cli_id, n)
        return r[n]

    def get_merged_recommendation(self, cli_id, n):
        print(f"Start computing recommendation system 1 (user-based) at: {datetime.now()}")
        r1 = self.rs1.get_recommendation(cli_id)
        print(f"Start computing recommendation system 2 (cluster-based-1) at: {datetime.now()}")
        r2 = self.rs2.get_recommendation(cli_id)
        print(f"Start computing recommendation system 3 (cluster-based-2) at: {datetime.now()}")
        r3 = self.rs3.get_recommendation(cli_id)

        print(f"Start computing the recommendation system merger at {datetime.now()}")

        def label_set(r):
            return set(map(lambda x: x["LIBELLE"], r))

        def score_dict(r):
            return {o["LIBELLE"]: i for i, o in enumerate(r)}

        # retrieve all label in common
        s = label_set(r1).intersection(label_set(r2), label_set(r3))
        score = {label: 0 for label in s}

        s1 = score_dict(r1)
        s2 = score_dict(r2)
        s3 = score_dict(r3)

        for label in score:
            score[label] = s1[label] + s2[label] + s3[label]

        result = list(sorted(score.items(), key=lambda x: x[1]))
        libelle = result[n][0]
        explanation = ""
        for r in r1:
            if r["LIBELLE"] == libelle:
                explanation += "USER_BASED: " + r["explanation"] + "\n"
                break
        for r in r2:
            if r["LIBELLE"] == libelle:
                explanation += "CLUSTER_BASED1: " + r["explanation"] + "\n"
                break
        for r in r3:
            if r["LIBELLE"] == libelle:
                explanation += "CLUSTER_BASED2: " + r["explanation"]
                break
        print(f"End computing at: {datetime.now()}")
        return {"LIBELLE": libelle, "explanation": explanation}


if __name__ == "__main__":
    merger = Merger()
    r = merger.get_recommendation(1490281, RecommendationType.ALL)
    print(dumps(r, indent=4))
    r = merger.get_recommendation(1490281, RecommendationType.CLUSTER_BASED_1)
    print(dumps(r, indent=4))
    r = merger.get_recommendation(1490281, RecommendationType.CLUSTER_BASED_2)
    print(dumps(r, indent=4))
    r = merger.get_recommendation(1490281, RecommendationType.USER_BASED)
    print(dumps(r, indent=4))
