def pretty_string_dict(d : dict, indent : int=0, digits : int=3, concatenate : bool=True):
    parts = []
    for k, v in d.items():
        if not isinstance(v, dict):
            if isinstance(v, float):
                part = f'{" " * indent}{k} : {v:.{digits}f}'
            else:
                part = f'{" " * indent}{k} : {v}'
            parts.append(part)
        else:
            part = f'{" " * indent}{k}:'
            parts.append(part)
            parts.extend(pretty_string_dict(v, indent=indent+2, concatenate=False))
    if concatenate:
        return "\n".join(parts)
    return parts