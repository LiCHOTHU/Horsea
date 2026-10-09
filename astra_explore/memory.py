"""Explicit, bounded interaction memory supplied to Astra (nothing is summarised by a second model)."""
import copy

RECENT = 6
PREVIOUS_ATTEMPTS = 2


class History:
    def __init__(self):
        self.interactions = []          # every executed (or stopped) decision, full records
        self.attempt_summaries = []     # one factual summary per finished attempt (+ Astra's final running memory)
        self.running = {"observations": [], "hypotheses": [], "summary": ""}   # Astra's own bounded memory
        self.next_id = 1

    def known_ids(self):
        return {r["id"] for r in self.interactions}

    def add_interaction(self, record):
        record = dict(record, id=self.next_id)
        self.next_id += 1
        self.interactions.append(record)
        return record["id"]

    def apply_memory_update(self, update):
        self.running = copy.deepcopy(update)

    def end_attempt(self, summary):
        self.attempt_summaries.append(dict(summary, astra_final_memory=copy.deepcopy(self.running)))

    def view(self):
        """What the model sees: the latest RECENT interactions in full, up to PREVIOUS_ATTEMPTS summaries, running memory."""
        return {"recent_interactions": copy.deepcopy(self.interactions[-RECENT:]),
                "older_interaction_ids_not_shown": [r["id"] for r in self.interactions[:-RECENT]],
                "previous_attempts": copy.deepcopy(self.attempt_summaries[-PREVIOUS_ATTEMPTS:]),
                "running_memory": copy.deepcopy(self.running)}

    def snapshot(self):
        return {"n_interactions": len(self.interactions), "attempt_summaries": copy.deepcopy(self.attempt_summaries),
                "running_memory": copy.deepcopy(self.running)}
