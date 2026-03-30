import numpy as np

class SmartRedistributor:
    def __init__(self, distance_weight=1.0, capacity_weight=1.5, urgency_weight=1.0):
        self.dw = distance_weight
        self.cw = capacity_weight
        self.uw = urgency_weight

    def compute_score(self, ngo):
        return (
            self.cw * ngo["capacity"]
            - self.dw * ngo["distance"]
            + self.uw * ngo.get("urgency", 0)
        )
    
    def allocate(self, surplus, ngos):

        for ngo in ngos:
            ngo["score"] = self.compute_score(ngo)

        ngos = sorted(ngos, key=lambda x: x["score"], reverse=True)

        allocation = []
        remaining = surplus

        for ngo in ngos:
            if remaining <= 0:
                break

            give = min(ngo["capacity"], remaining)

            allocation.append({
                "ngo": ngo["name"],
                "allocated": int(give),
                "distance": ngo["distance"]
            })

            remaining -= give

        return allocation, remaining