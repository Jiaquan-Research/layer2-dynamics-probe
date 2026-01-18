def parse_log_file(path):
    """
    Parse experiment logs into structured phase-space data.

    Returns:
        dict: regime_tag -> list of dicts
              { "step": int, "entropy": float, "consistency": float }
    """
    data = {"H": [], "L": [], "D": [], "G": []}

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("["):
                continue

            tag = line[1]
            if tag not in data:
                continue

            if "Cons: Init" in line:
                continue

            try:
                step = int(line.split("-")[1].split("]")[0])
                ent = float(line.split("Ent:")[1].split("|")[0])
                cons = float(line.split("Cons:")[1].split("|")[0])
            except Exception:
                continue

            data[tag].append({
                "step": step,
                "entropy": ent,
                "consistency": cons,
            })

    return data
