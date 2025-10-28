def extract_experience_name(
    experience_name: str, relative_csv_file_location: dict[str, str]
) -> str:
    """
    Extract the experience name from the experiment name.
    """
    for key, value in relative_csv_file_location.items():
        if key in experience_name:
            return value
    return None
