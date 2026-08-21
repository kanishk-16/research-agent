def clean_string(value):
    if value is None:
        return ""

    return str(value).strip()


def require_dict(value, field_name):
    if not isinstance(value, dict):
        raise RuntimeError(
            f"'{field_name}' must be a JSON object."
        )

    return value


def require_list(value, field_name):
    if not isinstance(value, list):
        raise RuntimeError(
            f"'{field_name}' must be a list."
        )

    return value


def clean_string_list(value, field_name):
    require_list(value, field_name)

    cleaned = []

    for item in value:
        text = clean_string(item)

        if text:
            cleaned.append(text)

    return cleaned


def require_bool(value, field_name):
    if not isinstance(value, bool):
        raise RuntimeError(
            f"'{field_name}' must be true or false."
        )

    return value


def require_non_negative_int(value, field_name):
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
    ):
        raise RuntimeError(
            f"'{field_name}' must be a non-negative integer."
        )

    return value


def require_positive_int(value, field_name):
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise RuntimeError(
            f"'{field_name}' must be a positive integer."
        )

    return value