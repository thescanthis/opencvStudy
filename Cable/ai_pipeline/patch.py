import re

with open('branching_wire_builder.py', 'r', encoding='utf-8') as f:
    code = f.read()

new_route = '''    def _route_wire_legs(self, paths, trunk_connector, from_connector, to_connector, uv_from, uv_to):
        legs = []
        if from_connector == to_connector:
            bp = paths.get(from_connector)
            if bp is None: return []
            legs.append((f"direct_{from_connector}_{uv_from}_{uv_to}", self._offset_path(bp, uv_from, uv_to)))
            return legs

        trunk_bp = paths.get(trunk_connector)
        if trunk_bp is None: return []

        if from_connector == trunk_connector:
            branch_bp = paths.get(to_connector)
            if branch_bp is None: return []
            leg1 = self._offset_path(trunk_bp, uv_from, (0.0, 0.0))
            leg2 = self._offset_path(branch_bp, (0.0, 0.0), uv_to, branch_only=True)
            legs.append((f"trunk_{from_connector}_{uv_from[0]:.3f}_{uv_from[1]:.3f}", leg1))
            legs.append((f"branch_{to_connector}_{uv_to[0]:.3f}_{uv_to[1]:.3f}_from_{uv_from[0]:.3f}_{uv_from[1]:.3f}", leg2))
            return legs

        if to_connector == trunk_connector:
            branch_bp = paths.get(from_connector)
            if branch_bp is None: return []
            leg1 = self._offset_path(branch_bp, uv_from, (0.0, 0.0), branch_only=True)
            leg2 = self._offset_path(trunk_bp, (0.0, 0.0), uv_to)
            legs.append((f"branch_{from_connector}_{uv_from[0]:.3f}_{uv_from[1]:.3f}_to_{uv_to[0]:.3f}_{uv_to[1]:.3f}", leg1))
            legs.append((f"trunk_{to_connector}_{uv_to[0]:.3f}_{uv_to[1]:.3f}", leg2))
            return legs

        bp_from = paths.get(from_connector)
        bp_to = paths.get(to_connector)
        if bp_from is None or bp_to is None: return []
        leg1 = self._offset_path(bp_from, uv_from, (0.0, 0.0), branch_only=True)[::-1]
        leg2 = self._offset_path(bp_to, (0.0, 0.0), uv_to, branch_only=True)
        legs.append((f"branch_{from_connector}_{uv_from[0]:.3f}_{uv_from[1]:.3f}_X", leg1))
        legs.append((f"branch_{to_connector}_{uv_to[0]:.3f}_{uv_to[1]:.3f}_X", leg2))
        return legs'''

code = re.sub(r'    def _route_wire\(self.*?return np\.concatenate\(\[leg1, leg2\], axis=0\)', new_route, code, flags=re.DOTALL)

old_loop = '''            path = self._route_wire(paths, trunk_connector, from_connector, to_connector, uv_from, uv_to)
            if path is None:
                skipped += 1
                continue

            mesh = create_tube_mesh(path, self.wire_radius, [255, 69, 0, 255])
            if len(mesh.vertices) > 0:
                meshes.append(mesh)'''

new_loop = '''            legs = self._route_wire_legs(paths, trunk_connector, from_connector, to_connector, uv_from, uv_to)
            if not legs:
                skipped += 1
                continue
            for key, path in legs:
                unique_legs[key] = path

        # Generate unique meshes
        for key, path in unique_legs.items():
            mesh = create_tube_mesh(path, self.wire_radius, [255, 69, 0, 255])
            if len(mesh.vertices) > 0:
                meshes.append(mesh)'''

code = code.replace(old_loop, new_loop)
code = code.replace('skipped = 0\n        for wire in topo.wires:', 'skipped = 0\n        unique_legs = {}\n        for wire in topo.wires:')

with open('branching_wire_builder.py', 'w', encoding='utf-8') as f:
    f.write(code)
