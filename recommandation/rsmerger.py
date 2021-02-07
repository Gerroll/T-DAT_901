from recommandation.systems import RSClusterBased1, RSClusterBased2, RSUserBased


class Merger:
    def __init__(self):
        self.rs1 = RSUserBased()
        self.rs2 = RSClusterBased1()
        self.rs3 = RSClusterBased2()

    def get_recommendation(self, cli_id):
        r1 = self.rs1.get_recommendation(cli_id)
        r2 = self.rs2.get_recommendation(cli_id)
        r3 = self.rs3.get_recommendation(cli_id)

        print(r1)
        print(r2)
        print(r3)


if __name__ == "__main__":
    merger = Merger()
    merger.get_recommendation(996899213)
