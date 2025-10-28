from typing import List, Optional

from abnumber import Chain
from torch.utils.data import DataLoader


def extract_cdr_positions_with_abnumber(
    predict_dataloader: DataLoader,
) -> Optional[List[List[int]]]:
    """
    Extract CDR positions using AbNumber library for accurate antibody numbering.

    Args:
        predict_dataloader: DataLoader containing batches with sequences.

    Returns:
        List of CDR positions or None if extraction fails.
    """
    cdrs = []
    cdr_names = ["CDR1", "CDR2", "CDR3"]
    for batch in predict_dataloader:
        for sequence in batch["sequence"]:
            chain = Chain(sequence, scheme="imgt", cdr_definition="imgt")
            positions = chain.positions
            cdr_pos = []
            for cdr in cdr_names:
                cdr_positions = [
                    pos for pos in positions.keys() if pos.get_region() == cdr
                ]
                if cdr_positions:
                    start = min(pos.number for pos in cdr_positions)
                    end = max(pos.number for pos in cdr_positions)
                else:
                    start, end = 0, 0
                cdr_pos.extend([start, end])
            cdrs.append(cdr_pos)
    return cdrs
