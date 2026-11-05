"""Output classes. The color is the side that GIVES the check: 'Check White' = the white
pieces attack the black king. A network stores its own labels: the analyzer reads them
from the .nn (the 6-class bonus network needs no specific code)."""

LABELS = ["Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black"]


def network_labels(net):
    if net.labels:
        return list(net.labels)
    if net.output_size == len(LABELS):
        return list(LABELS)
    raise ValueError(f"the network has {net.output_size} outputs and no labels: cannot name its predictions")


def label_index(label, labels):
    """'check  white' -> index of 'Check White' (case and spaces do not matter); ValueError if unknown."""
    wanted = " ".join(label.split()).lower()
    for i, name in enumerate(labels):
        if name.lower() == wanted:
            return i
    raise ValueError(f"unknown label '{label}' (expected one of: {', '.join(labels)})")
