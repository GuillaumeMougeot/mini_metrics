def pretty_string_dict(d : dict, indent : int=0, digits : int=3, concatenate : bool=True):
    parts = []
    for k, v in d.items():
        if not isinstance(v, dict):
            if isinstance(v, float):
                part = f'{" " * indent}{k}', f'{v:.{digits}f}'
            else:
                part = f'{" " * indent}{k}', f'{v}'
            parts.append(part)
        else:
            part = f'{" " * indent}{k}:'
            parts.append(part)
            parts.extend(pretty_string_dict(v, indent=indent+2, concatenate=False))
    first_row_width = max([len(f'{p[0]}') for p in parts if not isinstance(p, str)], default=0)
    parts = [p if isinstance(p, str) else f'{p[0]:<{first_row_width}} : {p[1]}' for p in parts]
    if concatenate:
        return "\n".join(parts)
    return parts