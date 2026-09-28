from pathlib import Path
import pandas as pd


def save_candidate_pairs(
    candidate_pairs,
    output_path: Path,
):
    """
    Save all generated candidate pairs.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidate_pairs.to_csv(
        output_path,
        sep="\t",
        index=False,
    )

    print(
        f"Candidate pairs saved to: "
        f"{output_path}"
    )


def save_matching_results(
    matches,
    output_path: Path,
):
    """
    Save final matching results.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    matches.to_csv(
        output_path,
        sep="\t",
        index=False,
    )

    print(
        f"Matching results saved to: "
        f"{output_path}"
    )


def create_matching_results(
    candidate_features,
    probabilities,
    threshold,
):
    """
    Select candidate pairs whose probability is
    greater than or equal to the selected threshold.
    """

    results = candidate_features[
        [
            "source1_index",
            "target_index",
        ]
    ].copy()

    results["match_probability"] = probabilities

    results = results[
        results["match_probability"] >= threshold
    ].copy()

    return results


def prepare_candidate_output(
    candidate_pairs,
    source_name,
):
    """
    Add source information to candidate pairs.
    """

    output = candidate_pairs.copy()

    output["target_source"] = source_name

    return output


def combine_candidate_pairs(
    source2_candidates,
    source3_candidates,
):
    """
    Combine candidates from Source 2 and Source 3.
    """

    return pd.concat(
        [
            source2_candidates,
            source3_candidates,
        ],
        ignore_index=True,
    )