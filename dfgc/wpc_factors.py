"""Our fixed WPC geometry/texture severity mapping; no MOS is embedded."""
WPC_GEOMETRY_SEVERITY = {'Downsampling': {9.0: 0.0, 8.0: 0.5, 7.0: 1.0}, 'G-PCC (T)/S-PCC': {8.0: 0.0, 6.0: 0.5, 4.0: 1.0}, 'Gaussian noise': {0.0: 0.0, 2.0: 0.5, 4.0: 1.0}, 'V-PCC': {1.0: 0.0, 2.0: 0.5, 3.0: 1.0}}
WPC_TEXTURE_SEVERITY = {'G-PCC (O)/L-PCC': {16.0: 0.0, 32.0: 1.0 / 3.0, 48.0: 2.0 / 3.0, 64.0: 1.0}, 'G-PCC (T)/S-PCC': {64.0: 0.0, 128.0: 1.0 / 3.0, 256.0: 2.0 / 3.0, 512.0: 1.0}, 'Gaussian noise': {8.0: 0.0, 16.0: 0.5, 32.0: 1.0}, 'V-PCC': {1.0: 0.0, 2.0: 0.5, 3.0: 1.0}}

def _wpc_factor_target(mapping, distortion_type: str, parameter: float):
    levels = mapping.get(distortion_type)
    if levels is None:
        return (0.0, 0.0)
    value = float(parameter)
    if value not in levels:
        raise ValueError(f'Unsupported WPC parameter for {distortion_type!r}: {value}; expected one of {sorted(levels)}.')
    return (float(levels[value]), 1.0)

def factor_metadata(distortion_type, geometry_parameter=None, texture_parameter=None):
    geometry, geometry_mask = _wpc_factor_target(WPC_GEOMETRY_SEVERITY, distortion_type, geometry_parameter)
    texture, texture_mask = _wpc_factor_target(WPC_TEXTURE_SEVERITY, distortion_type, texture_parameter)
    return dict(geometry_severity=geometry, geometry_severity_mask=geometry_mask,
                texture_severity=texture, texture_severity_mask=texture_mask)
