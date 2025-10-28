class NonCompliantDataframeError(Exception):
    """
    Only some of the columns_to_modify are present in your dataset which is problematic
    . Please ensure that either all the needed columns are here,or none of them.
    """
