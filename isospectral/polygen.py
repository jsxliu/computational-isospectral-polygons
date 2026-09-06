# This file generates simple closed polygons on the square lattice.
# Each side is parallel to an axis or a 45-degree diagonal, and its integer
# length is bounded by max_length. The search first chooses a direction
# pattern, then uses depth-first search to try signed edge lengths.
# Reachability and intersection checks discard partial paths that cannot become
# simple closed polygons. Canonical hashes identify and remove isometries.

import os
from .config import njit
import numpy as np

REACHABILITY_CACHE_DIR = os.path.join(".cache", "reachability")

# Codes 0, 1, 2, 3 mean horizontal, vertical, rising diagonal, and falling diagonal.
# The length's sign chooses the orientation. Consecutive sides cannot be
# parallel, so each code has three possible successors.
NEXT = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], dtype=np.int8)



# Takes all possible direction patterns (2*3^n) and indexes them from 0 to 2*3^n-1.
# Then each index is converted into base 3 and assigned a unique direction pattern.
@njit
def decode_direction_pattern(idx: np.int64, n: int) -> np.ndarray:
    s = np.empty(n, np.int8)
    if n == 0:
        return s

    total_rest = 1
    for _ in range(n - 1):
        total_rest *= 3

    top = idx // total_rest
    rem = idx - top * total_rest

    prev = np.int8(0 if top == 0 else 2)
    s[0] = prev

    if n == 1:
        return s

    digits = np.empty(n - 1, np.uint8)
    for i in range(n - 2, -1, -1):
        digits[i] = np.uint8(rem % 3)
        rem //= 3

    for i in range(1, n):
        prev = NEXT[prev, np.int64(digits[i - 1])]
        s[i] = prev

    return s


# Divides the set of direction patterns into chunks for parallel processing.
def split_range(total: int, parts: int):
    base, rem = divmod(total, parts)
    ranges = []
    start = 0
    for k in range(parts):
        sz = base + (1 if k < rem else 0)
        end = start + sz
        if start < end:
            ranges.append((start, end))
        start = end
    return ranges



# Use the shoelace theorem to compute the double the area of a polygon.
@njit
def _twice_area(coords, n):
    x = coords[:n, 0]
    y = coords[:n, 1]
    acc = np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    acc += x[-1] * y[0] - x[0] * y[-1]
    return abs(acc)


# Converts our direction code and a signed length into a coordinate displacement.
@njit
def _edge_displacement(code, length):
    if code == 0:
        return length, 0
    if code == 1:
        return 0, length
    if code == 2:
        return length, length
    if code == 3:
        return -length, length



# Self-intersecting check: 

#_orientation          determines which side of a line a point is, and if it is collinear.
#_on_segment           checks if a point lies on a line segment.
#_segments_intersect   combines the previous two tests to decide whether two edges intersect.

# Relative orientation of the ordered integer points a, b, c.
@njit
def _orientation(ax, ay, bx, by, cx, cy):
    v = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    return 1 if v > 0 else (-1 if v < 0 else 0)

# Given collinear a, b, c, test whether b lies on the closed segment ac.
@njit
def _on_segment(ax, ay, bx, by, cx, cy):
    return (min(ax, cx) <= bx <= max(ax, cx)) and (min(ay, cy) <= by <= max(ay, cy))

# Determine whether two line segments intersect.
# We do this by checking if each segment's endpoints lie on opposite
# sides of the other segment, or if they are collinear; i.e. an endpoint lies on the other segment.
@njit
def _segments_intersect(a1x, a1y, a2x, a2y, b1x, b1y, b2x, b2y):
    o1 = _orientation(a1x, a1y, a2x, a2y, b1x, b1y)
    o2 = _orientation(a1x, a1y, a2x, a2y, b2x, b2y)
    o3 = _orientation(b1x, b1y, b2x, b2y, a1x, a1y)
    o4 = _orientation(b1x, b1y, b2x, b2y, a2x, a2y)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(a1x, a1y, b1x, b1y, a2x, a2y):
        return True
    if o2 == 0 and _on_segment(a1x, a1y, b2x, b2y, a2x, a2y):
        return True
    if o3 == 0 and _on_segment(b1x, b1y, a1x, a1y, b2x, b2y):
        return True
    if o4 == 0 and _on_segment(b1x, b1y, a2x, a2y, b2x, b2y):
        return True
    return False



# Counts the direction types that remain at every position in a pattern.
@njit
def suffix_direction_counts(status_row: np.ndarray):
    n = status_row.shape[0]
    remaining = np.zeros((n + 1, 4), dtype=np.int16)

    for i in range(n - 1, -1, -1):
        for code in range(4):
            remaining[i, code] = remaining[i + 1, code]
        remaining[i, status_row[i]] += 1

    return remaining


# Depth-first search over signed edge lengths

# Explores edge lengths while only retaining paths that can form valid polygons.
# - crd: vertices of the current path
# - occ: lattice points already visited
# - d: index of the next edge
# - B: offset used to center coordinates in the reachability table
@njit
def _search_edge_lengths(status_row, allowed, positive,
            crd, survivors, n, survivor_count, d, max_survivors,
            occ, R, remaining_counts, reach, B):
    if d == n:
        # All edge lengths and directions are assigned, so we ensure a completed path closes and has positive area.
        if crd[n, 0] != 0 or crd[n, 1] != 0:
            return survivor_count
        x0, y0 = crd[0, 0], crd[0, 1]
        x1, y1 = crd[1, 0], crd[1, 1]
        x2, y2 = crd[n - 1, 0], crd[n - 1, 1]
        if (y1 - y0) * (x2 - x0) == (x1 - x0) * (y2 - y0):
            return survivor_count
        if _twice_area(crd, n) == 0:
            return survivor_count
        if survivor_count >= max_survivors:
            raise RuntimeError("polygon buffer is too small")
        survivors[survivor_count, :, :] = crd[:n]
        survivor_count += 1
        return survivor_count

    choices = positive if d == 0 else allowed
    px = crd[d, 0]
    py = crd[d, 1]
    code = status_row[d]

    for L in choices:
        dx, dy = _edge_displacement(code, L)
        nx = px + dx
        ny = py + dy

        # Reject a step when the remaining directions cannot close the path.
        r0 = remaining_counts[d + 1, 0]
        r1 = remaining_counts[d + 1, 1]
        r2 = remaining_counts[d + 1, 2]
        r3 = remaining_counts[d + 1, 3]
        ixx = B - nx
        iyy = B - ny
        if ixx < 0 or ixx >= reach.shape[4] or iyy < 0 or iyy >= reach.shape[5]:
            continue
        if not reach[r0, r1, r2, r3, iyy, ixx]:
            continue

        # Bounding boxes skip previous edges that are too far away.
        minx = px if px < nx else nx
        maxx = px if px > nx else nx
        miny = py if py < ny else ny
        maxy = py if py > ny else ny

        # A new edge may touch only its immediate predecessor (and, when
        # closing, the first edge at the origin).
        blocked_cross = False
        for k in range(d - 1):
            if d == n - 1 and nx == 0 and ny == 0 and k == 0:
                continue
            a1x = crd[k, 0]
            a1y = crd[k, 1]
            a2x = crd[k + 1, 0]
            a2y = crd[k + 1, 1]
            ea_minx = a1x if a1x < a2x else a2x
            ea_maxx = a1x if a1x > a2x else a2x
            ea_miny = a1y if a1y < a2y else a2y
            ea_maxy = a1y if a1y > a2y else a2y
            if ea_minx > maxx or ea_miny > maxy or ea_maxx < minx or ea_maxy < miny:
                continue
            if _segments_intersect(px, py, nx, ny, a1x, a1y, a2x, a2y):
                blocked_cross = True
                break
        if blocked_cross:
            continue

        # Reject edges that revisit a lattice point on the existing path (exceot the origin).
        sx = 0
        if dx > 0:
            sx = 1
        elif dx < 0:
            sx = -1
        sy = 0
        if dy > 0:
            sy = 1
        elif dy < 0:
            sy = -1
        steps = abs(dx)
        if abs(dy) > steps:
            steps = abs(dy)

        blocked = False
        x = px
        y = py
        for t in range(steps):
            x += sx
            y += sy
            ix2 = x + R
            iy2 = y + R
            if d == n - 1 and t == steps - 1 and x == 0 and y == 0:
                pass
            else:
                if occ[iy2, ix2] != 0:
                    blocked = True
                    break
        if blocked:
            continue

        # Mark the new edge, recurse, then undo the marks for the next choice.
        x = px
        y = py
        for t in range(steps):
            x += sx
            y += sy
            occ[y + R, x + R] = 1

        crd[d + 1, 0] = nx
        crd[d + 1, 1] = ny

        survivor_count = _search_edge_lengths(
            status_row, allowed, positive,
            crd, survivors, n, survivor_count, d + 1, max_survivors,
            occ, R, remaining_counts, reach, B,
        )

        x = px
        y = py
        for t in range(steps):
            x += sx
            y += sy
            occ[y + R, x + R] = 0

    return survivor_count


# Produces and stores all valid polygons associated with one fixed direction pattern.
# Uses the recursion in _search_edge_lengths to manage the search.
# max_survivors caps stored polygons per pattern to bound memory use.
@njit
def generate_polygons_for_pattern(status_row, max_len, allowed, positive,
                                  remaining_counts, reach, B,
                                  max_survivors=100000):
    n = status_row.shape[0]
    # First and last sides meet at the origin. Parallel directions there
    # would form a redundant 180-degree vertex and are always rejected.
    if n > 1 and status_row[0] == status_row[n - 1]:
        return np.empty((0, n, 2), np.int32)
    crd = np.zeros((n + 1, 2), np.int32)
    R = n * max_len + 2
    G = 2 * R + 1
    occ = np.zeros((G, G), np.uint8)
    occ[R, R] = 1
    survivors = np.zeros((max_survivors, n, 2), np.int32)
    count = _search_edge_lengths(status_row, allowed, positive,
                    crd, survivors, n, 0, 0, max_survivors,
                    occ, R, remaining_counts, reach, B)
    return survivors[:count]


# Gives congruent polygons the same hash for duplicate polygon removal.
@njit
def canonical_hashes(survivors):
    count = survivors.shape[0]
    if count == 0:
        return np.empty(0, np.uint64), np.empty(0, np.uint64)

    n = survivors.shape[1]
    h1_out = np.empty(count, dtype=np.uint64)
    h2_out = np.empty(count, dtype=np.uint64)

# Reuse temporary arrays when processing each polygon.
    dx0 = np.empty(n, np.int8)
    dy0 = np.empty(n, np.int8)
    dx = np.empty(n, np.int8)
    dy = np.empty(n, np.int8)
    rdx = np.empty(n, np.int8)
    rdy = np.empty(n, np.int8)

# Constants used to initialize and convert the polygon’s edge sequence into a 64-bit hash.
    FNV_OFF = np.uint64(1469598103934665603)
    FNV_PRM = np.uint64(1099511628211)

    for p in range(count):
        poly = survivors[p]
        for i in range(n - 1):
            dx0[i] = np.int8(poly[i + 1, 0] - poly[i, 0])
            dy0[i] = np.int8(poly[i + 1, 1] - poly[i, 1])
        dx0[n - 1] = np.int8(poly[0, 0] - poly[n - 1, 0])
        dy0[n - 1] = np.int8(poly[0, 1] - poly[n - 1, 1])

        best_h1 = np.uint64(0xffffffffffffffff)
        best_h2 = np.uint64(0xffffffffffffffff)
        # All eight symmetries of the square: four rotations and four reflections.
        for t in range(8):
            for i in range(n):
                x = dx0[i]
                y = dy0[i]
                if t == 0:   dx[i] = x;        dy[i] = y
                elif t == 1: dx[i] = np.int8(-y); dy[i] = x
                elif t == 2: dx[i] = np.int8(-x); dy[i] = np.int8(-y)
                elif t == 3: dx[i] = y;        dy[i] = np.int8(-x)
                elif t == 4: dx[i] = x;        dy[i] = np.int8(-y)
                elif t == 5: dx[i] = np.int8(-x); dy[i] = y
                elif t == 6: dx[i] = y;        dy[i] = x
                else:        dx[i] = np.int8(-y); dy[i] = np.int8(-x)
            # Chooses a canonical starting edge for the polygon
            best_rot = 0
            for k in range(1, n):
                for j in range(n):
                    a_dx = dx[(best_rot + j) % n]
                    a_dy = dy[(best_rot + j) % n]
                    b_dx = dx[(k + j) % n]
                    b_dy = dy[(k + j) % n]
                    # Gives the lexicographically smallest cyclic edge sequence
                    if a_dx < b_dx:
                        break
                    if a_dx > b_dx:
                        best_rot = k
                        break
                    if a_dy < b_dy:
                        break
                    if a_dy > b_dy:
                        best_rot = k
                        break
            # Hash the lexicographically smallest edge sequence for this symmetry.
            h = FNV_OFF
            for i in range(n):
                b1 = np.uint8(dx[(best_rot + i) % n] + 3)
                b2 = np.uint8(dy[(best_rot + i) % n] + 3)
                h ^= np.uint64(b1)
                h *= FNV_PRM
                h ^= np.uint64(b2)
                h *= FNV_PRM
            fwd_h = h
            # Create the edge sequence for traversing the polygon backward.
            for i in range(n):
                rdx[i] = np.int8(-dx[n - 1 - i])
                rdy[i] = np.int8(-dy[n - 1 - i])
            # Chooses a canonical starting edge for the reverse polygon
            best_rot_r = 0
            for k in range(1, n):
                for j in range(n):
                    a_dx = rdx[(best_rot_r + j) % n]
                    a_dy = rdy[(best_rot_r + j) % n]
                    b_dx = rdx[(k + j) % n]
                    b_dy = rdy[(k + j) % n]
                    # Gives the lexicographically smallest cyclic edge sequence
                    if a_dx < b_dx:
                        break
                    if a_dx > b_dx:
                        best_rot_r = k
                        break
                    if a_dy < b_dy:
                        break
                    if a_dy > b_dy:
                        best_rot_r = k
                        break
            # Hash the lexicographically smallest edge sequence for the reverse symmetry.
            h = FNV_OFF
            for i in range(n):
                b1 = np.uint8(rdx[(best_rot_r + i) % n] + 3)
                b2 = np.uint8(rdy[(best_rot_r + i) % n] + 3)
                h ^= np.uint64(b1)
                h *= FNV_PRM
                h ^= np.uint64(b2)
                h *= FNV_PRM
            rev_h = h
            # Store the forward and reverse hashes in order depending on which is smaller.
            if fwd_h < rev_h:
                a1, a2 = fwd_h, rev_h
            else:
                a1, a2 = rev_h, fwd_h

            if a1 < best_h1 or (a1 == best_h1 and a2 < best_h2):
                best_h1, best_h2 = a1, a2

        h1_out[p] = best_h1
        h2_out[p] = best_h2

    return h1_out, h2_out


# Computes the exact area and perimeter to group before spectral screening.
@njit
def polygon_invariants_batch(survivors):
    count = survivors.shape[0]
    n = survivors.shape[1]
    a2_out = np.empty(count, dtype=np.int64)
    axial_out = np.empty(count, dtype=np.int64)
    diagonal_out = np.empty(count, dtype=np.int64)
    for p in range(count):
        a2 = np.int64(0)
        axial = np.int64(0)
        diagonal = np.int64(0)
        # Traverse every edge, computing a running total for area and perimeter.
        for i in range(n):
            j = (i + 1) % n
            xi = np.int64(survivors[p, i, 0])
            yi = np.int64(survivors[p, i, 1])
            xj = np.int64(survivors[p, j, 0])
            yj = np.int64(survivors[p, j, 1])
            a2 += xi * yj - xj * yi
            dx = xj - xi
            dy = yj - yi
            adx = abs(dx)
            ady = abs(dy)
            if adx == 0 or ady == 0:
                axial += adx + ady
            else:
                diagonal += adx
        if a2 < 0:
            a2 = -a2
        a2_out[p] = a2
        axial_out[p] = axial
        diagonal_out[p] = diagonal
    return a2_out, axial_out, diagonal_out


# Adds possible edge displacements to every currently reachable endpoint.
def _shift_or(src: np.ndarray, dst: np.ndarray, dx: int, dy: int):
    G = src.shape[1]
    if dx >= 0:
        sx0, sx1 = 0, G - dx
        dx0, dx1 = dx, G
    else:
        sx0, sx1 = -dx, G
        dx0, dx1 = 0, G + dx
    if dy >= 0:
        sy0, sy1 = 0, G - dy
        dy0, dy1 = dy, G
    else:
        sy0, sy1 = -dy, G
        dy0, dy1 = 0, G + dy
    if sx0 < sx1 and sy0 < sy1:
        dst[dy0:dy1, dx0:dx1] |= src[sy0:sy1, sx0:sx1]


# Loads or builds the endpoint table that rules out paths unable to close.
def ensure_reachability_cache(n: int, max_len: int):
    B = n * max_len
    G = 2 * B + 1
    os.makedirs(REACHABILITY_CACHE_DIR, exist_ok=True)
    fname = os.path.join(
        REACHABILITY_CACHE_DIR, f"reach_cache_{n}_{max_len}.npy"
    )
    if not os.path.exists(fname):
        print(f"[precompute] building reach cache {fname} …")
        reach = np.zeros((n + 1, n + 1, n + 1, n + 1, G, G), dtype=np.bool_)
        center = B
        reach[0, 0, 0, 0, center, center] = True

        shifts = [
            [(+L, 0) for L in range(1, max_len + 1)]
            + [(-L, 0) for L in range(1, max_len + 1)],
            [(0, +L) for L in range(1, max_len + 1)]
            + [(0, -L) for L in range(1, max_len + 1)],
            [(+L, +L) for L in range(1, max_len + 1)]
            + [(-L, -L) for L in range(1, max_len + 1)],
            [(-L, +L) for L in range(1, max_len + 1)]
            + [(+L, -L) for L in range(1, max_len + 1)],
        ]

        for total in range(0, n):
            for k0 in range(0, total + 1):
                for k1 in range(0, total - k0 + 1):
                    for k2 in range(0, total - k0 - k1 + 1):
                        k3 = total - k0 - k1 - k2
                        cur = reach[k0, k1, k2, k3]
                        if not cur.any():
                            continue
                        for code in range(4):
                            ks = [k0, k1, k2, k3]
                            ks[code] += 1
                            if ks[code] <= n:
                                dst = reach[ks[0], ks[1], ks[2], ks[3]]
                                for ddx, ddy in shifts[code]:
                                    _shift_or(cur, dst, ddx, ddy)

        np.save(fname, reach)
        print(f"[precompute] saved {fname}")
    return fname, B
