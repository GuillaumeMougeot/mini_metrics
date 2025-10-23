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

def format_table(d : dict, keys : list[str] | tuple[str, ...], digits : int=2):
    if isinstance(list(d.values())[0], (float, int)):
        ds : dict[str, dict[int, float]] = {k : {0 : float(v)} for k, v in d.items() if k in keys}
    else:
        ds : dict[str, dict[int, float]] = {k : v for k, v in d.items() if k in keys}
    rows = set([tuple(id.keys()) for id in ds.values()])
    if len(rows) != 1:
        raise RuntimeError(f'Inner dictionaries contain different keys: {rows}')
    rows = list(rows)[0]
    cols = list(ds.keys())
    lines = [[""] + cols]
    for row in rows:
        lines.append([f'level {row}'] + [f'{ds[col][row]:.{digits}%}' for col in cols])
    col_widths = [max([len(line[c]) + 2 for line in lines]) for c in range(len(cols)+1)]
    fmt_row = f'{{:>{col_widths[0]}}} | ' + " | ".join([f'{{:^{cw}}}' for cw in col_widths[1:]])
    lines = [fmt_row.format(*line) for line in lines]
    divider = "-|-".join(["-" * cw for cw in col_widths])
    lines.insert(1, divider)
    if len(rows) == 1:
        lines = [line[(col_widths[0] + 2):] for line in lines]
    return "\n".join(lines)

