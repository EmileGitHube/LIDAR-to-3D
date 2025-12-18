bl_info = {
    "name": "Lidar Pro V17 - Vectorial Geometric Buildings",
    "author": "Claude Assistant - Vectorial Version",
    "version": (17, 0, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Lidar Pro",
    "description": "Import LIDAR + Vectorial Geometric Buildings (Alpha Shape + DBSCAN) + GeoNodes Vegetation",
    "category": "Object",
}

import bpy
import bmesh
import numpy as np
import time
import random
import os
import subprocess
import sys
import math
import webbrowser
from mathutils import Vector, geometry
from mathutils.bvhtree import BVHTree
from bpy_extras.io_utils import ImportHelper

# ==============================================================================
# CONFIGURATION SYSTEME
# ==============================================================================
sys.setrecursionlimit(20000)

addon_dir = os.path.dirname(os.path.realpath(__file__))
modules_dir = os.path.join(addon_dir, "modules")
if modules_dir not in sys.path:
    sys.path.append(modules_dir)

# ==============================================================================
# DEPENDANCES
# ==============================================================================
def install_laspy():
    python_exe = sys.executable
    try:
        subprocess.check_call([python_exe, "-m", "pip", "install", "laspy[lazrs]", "--user"])
        return True
    except subprocess.CalledProcessError:
        return False

def check_laspy_installed():
    try:
        import laspy
        return True
    except ImportError:
        return False

# ==============================================================================
# ASSETS
# ==============================================================================
class AssetLibraryManager:
    @staticmethod
    def get_library_path():
        addon_dir = os.path.dirname(os.path.realpath(__file__))
        library_path = os.path.join(addon_dir, "assets", "library.blend")
        return library_path if os.path.exists(library_path) else None

    @staticmethod
    def link_object(object_name, link=True):
        if object_name in bpy.data.objects: return bpy.data.objects[object_name]
        library_path = AssetLibraryManager.get_library_path()
        if not library_path: return None
        try:
            with bpy.data.libraries.load(library_path, link=link) as (data_from, data_to):
                if object_name in data_from.objects: data_to.objects = [object_name]
            if data_to.objects:
                obj = data_to.objects[0]
                if link and obj.library:
                    obj = obj.copy()
                    obj.data = obj.data.copy()
                    if obj.name not in bpy.context.scene.objects:
                        bpy.context.scene.collection.objects.link(obj)
                return obj
        except Exception: pass
        return None

    @staticmethod
    def link_material(material_name):
        for mat in bpy.data.materials:
            if mat.name.lower() == material_name.lower(): return mat
        library_path = AssetLibraryManager.get_library_path()
        if not library_path: return None
        try:
            with bpy.data.libraries.load(library_path, link=False) as (data_from, data_to):
                found_name = None
                for m in data_from.materials:
                    if m.lower() == material_name.lower():
                        found_name = m; break
                if found_name: data_to.materials = [found_name]
            return data_to.materials[0] if data_to.materials else None
        except Exception: return None

# ==============================================================================
# CREATION MATERIAUX PROCEDURAUX
# ==============================================================================
def create_roof_material(name, intensity_avg, is_dark):
    """Créer un matériau de toit procédural (tuiles ou bac acier)"""
    mat = bpy.data.materials.get(name)
    if mat:
        return mat

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (400, 0)

    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    if is_dark:  # TUILES
        coord = nodes.new('ShaderNodeTexCoord')
        coord.location = (-800, 0)

        mapping = nodes.new('ShaderNodeMapping')
        mapping.location = (-600, 0)
        mapping.inputs['Scale'].default_value = (4.0, 2.0, 1.0)
        links.new(coord.outputs['UV'], mapping.inputs['Vector'])

        noise1 = nodes.new('ShaderNodeTexNoise')
        noise1.location = (-400, 100)
        noise1.inputs['Scale'].default_value = 15.0
        noise1.inputs['Detail'].default_value = 8.0
        links.new(mapping.outputs['Vector'], noise1.inputs['Vector'])

        voronoi = nodes.new('ShaderNodeTexVoronoi')
        voronoi.location = (-400, -100)
        voronoi.inputs['Scale'].default_value = 8.0
        voronoi.voronoi_dimensions = '2D'
        voronoi.feature = 'DISTANCE_TO_EDGE'
        links.new(mapping.outputs['Vector'], voronoi.inputs['Vector'])

        mix_relief = nodes.new('ShaderNodeMixRGB')
        mix_relief.location = (-200, 0)
        mix_relief.blend_type = 'MULTIPLY'
        mix_relief.inputs['Fac'].default_value = 0.5
        links.new(noise1.outputs['Color'], mix_relief.inputs['Color1'])
        links.new(voronoi.outputs['Distance'], mix_relief.inputs['Color2'])

        ramp = nodes.new('ShaderNodeValToRGB')
        ramp.location = (-200, -200)
        ramp.color_ramp.elements[0].position = 0.45
        ramp.color_ramp.elements[1].position = 0.55

        val = 0.15 + intensity_avg * 0.25
        ramp.color_ramp.elements[0].color = (val*0.8, val*0.4, val*0.3, 1.0)
        ramp.color_ramp.elements[1].color = (val*1.2, val*0.6, val*0.4, 1.0)

        links.new(voronoi.outputs['Distance'], ramp.inputs['Fac'])
        links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])
        links.new(mix_relief.outputs['Color'], bsdf.inputs['Roughness'])

        bsdf.inputs['Roughness'].default_value = 0.9
        bsdf.inputs['Specular IOR Level'].default_value = 0.3

    else:  # BAC ACIER
        coord = nodes.new('ShaderNodeTexCoord')
        coord.location = (-600, 0)

        mapping = nodes.new('ShaderNodeMapping')
        mapping.location = (-400, 0)
        mapping.inputs['Scale'].default_value = (0.5, 8.0, 1.0)
        links.new(coord.outputs['UV'], mapping.inputs['Vector'])

        wave = nodes.new('ShaderNodeTexWave')
        wave.location = (-200, 0)
        wave.wave_type = 'BANDS'
        wave.bands_direction = 'Y'
        wave.inputs['Scale'].default_value = 10.0
        wave.inputs['Distortion'].default_value = 0.5
        links.new(mapping.outputs['Vector'], wave.inputs['Vector'])

        noise = nodes.new('ShaderNodeTexNoise')
        noise.location = (-200, -200)
        noise.inputs['Scale'].default_value = 50.0
        noise.inputs['Detail'].default_value = 4.0
        links.new(coord.outputs['UV'], noise.inputs['Vector'])

        gray_val = 0.6 + intensity_avg * 0.3

        color_base = nodes.new('ShaderNodeRGB')
        color_base.location = (-200, 100)
        color_base.outputs[0].default_value = (gray_val, gray_val, gray_val*1.05, 1.0)

        mix = nodes.new('ShaderNodeMixRGB')
        mix.location = (0, 100)
        mix.blend_type = 'OVERLAY'
        mix.inputs['Fac'].default_value = 0.1
        links.new(color_base.outputs['Color'], mix.inputs['Color1'])
        links.new(noise.outputs['Color'], mix.inputs['Color2'])

        links.new(mix.outputs['Color'], bsdf.inputs['Base Color'])
        links.new(wave.outputs['Color'], bsdf.inputs['Roughness'])

        bsdf.inputs['Metallic'].default_value = 0.7
        bsdf.inputs['Roughness'].default_value = 0.4
        bsdf.inputs['Specular IOR Level'].default_value = 0.5

    return mat

def create_wall_material(name, gray_factor):
    """Créer un matériau de mur procédural (blanc cassé à gris)"""
    mat = bpy.data.materials.get(name)
    if mat:
        return mat

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (400, 0)

    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    coord = nodes.new('ShaderNodeTexCoord')
    coord.location = (-600, 0)

    noise1 = nodes.new('ShaderNodeTexNoise')
    noise1.location = (-400, 100)
    noise1.inputs['Scale'].default_value = 25.0
    noise1.inputs['Detail'].default_value = 10.0
    links.new(coord.outputs['UV'], noise1.inputs['Vector'])

    noise2 = nodes.new('ShaderNodeTexNoise')
    noise2.location = (-400, -100)
    noise2.inputs['Scale'].default_value = 5.0
    noise2.inputs['Detail'].default_value = 3.0
    links.new(coord.outputs['UV'], noise2.inputs['Vector'])

    base_val = 0.85 - (gray_factor * 0.35)

    color_base = nodes.new('ShaderNodeRGB')
    color_base.location = (-200, 200)
    color_base.outputs[0].default_value = (base_val, base_val*0.98, base_val*0.96, 1.0)

    mix1 = nodes.new('ShaderNodeMixRGB')
    mix1.location = (-200, 0)
    mix1.blend_type = 'OVERLAY'
    mix1.inputs['Fac'].default_value = 0.15
    links.new(color_base.outputs['Color'], mix1.inputs['Color1'])
    links.new(noise1.outputs['Color'], mix1.inputs['Color2'])

    mix2 = nodes.new('ShaderNodeMixRGB')
    mix2.location = (0, 100)
    mix2.blend_type = 'MULTIPLY'
    mix2.inputs['Fac'].default_value = 0.2
    links.new(mix1.outputs['Color'], mix2.inputs['Color1'])
    links.new(noise2.outputs['Color'], mix2.inputs['Color2'])

    links.new(mix2.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(noise1.outputs['Fac'], bsdf.inputs['Roughness'])

    bsdf.inputs['Roughness'].default_value = 0.85
    bsdf.inputs['Specular IOR Level'].default_value = 0.2

    return mat

# ==============================================================================
# ALGORITHMES OPTIMISES - PURE NUMPY
# ==============================================================================

# ==============================================================================
# ALGORITHMES GEOMETRIQUES POUR BATIMENTS (VECTORIEL)
# ==============================================================================

def filter_facade_points(points_xyz, k_neighbors=10, variance_threshold=3.0):
    """
    Filtre les points de façade en détectant la verticalité (KDTree optimisé).
    Si variance_z >> variance_xy, c'est une façade.

    Args:
        points_xyz: array (N, 3) de coordonnées XYZ
        k_neighbors: nombre de voisins à analyser
        variance_threshold: ratio variance_z/variance_xy pour détecter une façade

    Returns:
        mask: array (N,) bool, True = point de TOIT (à garder)
    """
    from mathutils.kdtree import KDTree

    if len(points_xyz) < k_neighbors:
        return np.ones(len(points_xyz), dtype=bool)

    # Construire KDTree sur XY uniquement (Z=0 pour recherche 2D)
    kd = KDTree(len(points_xyz))
    for i, pt in enumerate(points_xyz):
        kd.insert((pt[0], pt[1], 0.0), i)
    kd.balance()

    is_roof = np.ones(len(points_xyz), dtype=bool)

    # Échantillonnage intelligent : max 3000 points analysés
    sample_step = max(1, len(points_xyz) // 3000)
    sample_indices = np.arange(0, len(points_xyz), sample_step)

    for i in sample_indices:
        # Trouver les k voisins proches avec KDTree (ultra rapide - O(log N))
        k_actual = min(k_neighbors, len(points_xyz))
        neighbors_data = kd.find_n((points_xyz[i, 0], points_xyz[i, 1], 0.0), k_actual)

        if len(neighbors_data) < 3:
            continue

        # Extraire les indices des voisins
        neighbor_indices = [idx for (co, idx, dist) in neighbors_data]
        neighbors = points_xyz[neighbor_indices]

        # Calculer variance en Z et en XY
        var_z = np.var(neighbors[:, 2])
        var_xy = np.var(neighbors[:, 0]) + np.var(neighbors[:, 1])

        # Si variance Z très grande par rapport à XY -> façade
        if var_xy > 0 and (var_z / var_xy) > variance_threshold:
            is_roof[i] = False
            # Marquer aussi les voisins comme façade
            for idx in neighbor_indices:
                is_roof[idx] = False

    return is_roof

def dbscan_kdtree(points_2d, eps=1.5, min_samples=10, kd_tree=None):
    """
    DBSCAN optimisé avec KDTree de Blender (mathutils).
    Complexité: O(N log N) au lieu de O(N²).

    Args:
        points_2d: array (N, 2) de coordonnées XY
        eps: distance maximale entre voisins
        min_samples: nombre min de points pour former un cluster
        kd_tree: KDTree préconstruit (optionnel, sinon créé automatiquement)

    Returns:
        labels: array (N,) d'entiers, -1 = bruit, >= 0 = ID du cluster
    """
    from mathutils.kdtree import KDTree

    n = len(points_2d)
    labels = -np.ones(n, dtype=np.int32)
    visited = np.zeros(n, dtype=bool)

    # Construire KDTree si non fourni
    if kd_tree is None:
        kd = KDTree(n)
        for i, pt in enumerate(points_2d):
            kd.insert((pt[0], pt[1], 0.0), i)
        kd.balance()
    else:
        kd = kd_tree

    cluster_id = 0

    for i in range(n):
        if visited[i]:
            continue

        visited[i] = True

        # Trouver voisins dans le rayon eps avec KDTree (O(log N))
        neighbors_data = kd.find_range((points_2d[i, 0], points_2d[i, 1], 0.0), eps)
        neighbor_indices = [idx for (co, idx, dist) in neighbors_data]

        if len(neighbor_indices) < min_samples:
            labels[i] = -1  # Bruit
            continue

        # Créer un nouveau cluster
        labels[i] = cluster_id

        # Expansion du cluster (Flood Fill avec stack)
        stack = list(neighbor_indices)

        while stack:
            q = stack.pop()

            if visited[q]:
                continue

            visited[q] = True

            # Trouver voisins de q avec KDTree
            neighbors_q_data = kd.find_range((points_2d[q, 0], points_2d[q, 1], 0.0), eps)
            neighbors_q = [idx for (co, idx, dist) in neighbors_q_data]

            if len(neighbors_q) >= min_samples:
                for nq in neighbors_q:
                    if not visited[nq]:
                        stack.append(nq)

            if labels[q] == -1:
                labels[q] = cluster_id

        cluster_id += 1

    return labels

def convex_hull_numpy(points):
    """Calcule le Convex Hull en NumPy pur (Graham Scan)"""
    if len(points) < 3:
        return points

    # Trouver le point le plus bas (et le plus à gauche si égalité)
    points = np.array(points)
    start_idx = np.lexsort((points[:, 0], points[:, 1]))[0]
    start = points[start_idx]

    # Trier par angle polaire
    def polar_angle(p):
        dx = p[0] - start[0]
        dy = p[1] - start[1]
        return np.arctan2(dy, dx)

    sorted_points = sorted(points, key=polar_angle)

    # Graham Scan
    hull = [sorted_points[0], sorted_points[1]]

    for p in sorted_points[2:]:
        while len(hull) > 1:
            # Test de rotation (produit vectoriel)
            o = hull[-2]
            a = hull[-1]
            cross = (a[0] - o[0]) * (p[1] - o[1]) - (a[1] - o[1]) * (p[0] - o[0])
            if cross > 0:
                break
            hull.pop()
        hull.append(p)

    return hull

def alpha_shape_2d(points_2d, alpha=3.0):
    """
    Calcule l'Alpha Shape (Concave Hull) 2D en NumPy pur.
    Méthode : Delaunay Triangulation + filtrage des arêtes > alpha.

    Args:
        points_2d: array (N, 2) de coordonnées XY
        alpha: paramètre de concavité (plus petit = plus concave)

    Returns:
        boundary_points: liste de (x, y) formant le contour ordonné
    """
    if len(points_2d) < 4:
        return points_2d.tolist()

    # Delaunay Triangulation
    vecs = [Vector((p[0], p[1])) for p in points_2d]
    try:
        res = geometry.delaunay_2d_cdt(vecs, [], [], 0, 0.001)
        verts, faces = res[0], res[2]
    except:
        # Fallback sur convex hull
        return convex_hull_numpy(points_2d)

    # Filtrer les arêtes selon alpha
    edges_dict = {}
    alpha_sq = alpha * alpha

    for face in faces:
        for i in range(3):
            v1_idx = face[i]
            v2_idx = face[(i + 1) % 3]

            v1 = verts[v1_idx]
            v2 = verts[v2_idx]

            edge_length_sq = (v1 - v2).length_squared

            # Garder seulement les arêtes courtes
            if edge_length_sq <= alpha_sq:
                edge_key = tuple(sorted([v1_idx, v2_idx]))
                edges_dict[edge_key] = edges_dict.get(edge_key, 0) + 1

    # Extraire les arêtes de bordure (comptées 1 fois)
    boundary_edges = [k for k, v in edges_dict.items() if v == 1]

    if not boundary_edges:
        # Fallback: convex hull NumPy pur
        return convex_hull_numpy(points_2d)

    # Reconstruire le contour ordonné
    adjacency = {}
    for v1, v2 in boundary_edges:
        adjacency.setdefault(v1, []).append(v2)
        adjacency.setdefault(v2, []).append(v1)

    # Parcours du contour
    start = boundary_edges[0][0]
    boundary = [start]
    visited = {start}
    current = start

    while len(boundary) < len(adjacency):
        neighbors = [n for n in adjacency.get(current, []) if n not in visited]
        if not neighbors:
            break
        current = neighbors[0]
        visited.add(current)
        boundary.append(current)

    # Convertir en coordonnées
    boundary_coords = [(verts[i].x, verts[i].y) for i in boundary]

    return boundary_coords

def rectify_polygon_to_right_angles(polygon_2d, angle_tolerance=15.0):
    """
    Régularise un polygone pour favoriser les angles droits (90°).

    Méthode:
    1. Trouve l'orientation principale (axe du bounding box minimal)
    2. Rotate le polygone pour aligner sur X/Y
    3. Snap les coordonnées sur une grille fine
    4. Simplifie les segments alignés
    5. Rotate inverse

    Args:
        polygon_2d: liste de (x, y)
        angle_tolerance: tolérance angulaire en degrés

    Returns:
        rectified_polygon: liste de (x, y) régularisé
    """
    if len(polygon_2d) < 3:
        return polygon_2d

    points = np.array(polygon_2d)

    # 1. Trouver orientation principale via PCA simple
    centroid = points.mean(axis=0)
    centered = points - centroid

    # Covariance matrix
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # Principal axis
    principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
    angle = np.arctan2(principal_axis[1], principal_axis[0])

    # 2. Rotation pour aligner sur axes
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    rotated = (rotation_matrix @ centered.T).T

    # 3. Snap sur grille 0.1m
    grid_size = 0.1
    snapped = np.round(rotated / grid_size) * grid_size

    # 4. Simplification : fusionner les points proches
    simplified = [snapped[0]]
    for i in range(1, len(snapped)):
        dist = np.linalg.norm(snapped[i] - simplified[-1])
        if dist > 0.3:  # Seuil de fusion
            simplified.append(snapped[i])

    simplified = np.array(simplified)

    # 5. Rotation inverse
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rotation_matrix_inv = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    final = (rotation_matrix_inv @ simplified.T).T + centroid

    return final.tolist()

def rasterize_points_to_grid(points_2d, resolution=0.5):
    """
    Projette les points 2D sur une grille binaire.
    Args:
        points_2d: array (N, 2) de coordonnées XY
        resolution: taille de cellule en mètres
    Returns:
        grid: grille binaire 2D (bool)
        origin: (min_x, min_y) de la grille
        cell_size: taille de cellule
    """
    if len(points_2d) == 0:
        return None, None, resolution

    min_x, min_y = points_2d[:, 0].min(), points_2d[:, 1].min()
    max_x, max_y = points_2d[:, 0].max(), points_2d[:, 1].max()

    # Calculer la taille de la grille
    width = int(np.ceil((max_x - min_x) / resolution)) + 1
    height = int(np.ceil((max_y - min_y) / resolution)) + 1

    # Créer la grille vide
    grid = np.zeros((height, width), dtype=bool)

    # Convertir les points en indices de grille
    px = ((points_2d[:, 0] - min_x) / resolution).astype(np.int32)
    py = ((points_2d[:, 1] - min_y) / resolution).astype(np.int32)

    # Clamp pour éviter les débordements
    px = np.clip(px, 0, width - 1)
    py = np.clip(py, 0, height - 1)

    # Marquer les cellules occupées
    grid[py, px] = True

    return grid, (min_x, min_y), resolution

def binary_dilation(grid, iterations=1):
    """Dilatation binaire (morphologie) en NumPy pur"""
    result = grid.copy()
    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=bool)

    for _ in range(iterations):
        h, w = result.shape
        dilated = np.zeros_like(result)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if kernel[dy + 1, dx + 1]:
                    shifted = np.zeros_like(result)
                    y_start = max(0, -dy)
                    y_end = min(h, h - dy)
                    x_start = max(0, -dx)
                    x_end = min(w, w - dx)

                    shifted[y_start + dy:y_end + dy, x_start + dx:x_end + dx] = \
                        result[y_start:y_end, x_start:x_end]
                    dilated |= shifted
        result = dilated
    return result

def binary_erosion(grid, iterations=1):
    """Érosion binaire (morphologie) en NumPy pur"""
    result = grid.copy()
    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=bool)

    for _ in range(iterations):
        h, w = result.shape
        eroded = np.ones_like(result)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if kernel[dy + 1, dx + 1]:
                    shifted = np.zeros_like(result)
                    y_start = max(0, -dy)
                    y_end = min(h, h - dy)
                    x_start = max(0, -dx)
                    x_end = min(w, w - dx)

                    shifted[y_start + dy:y_end + dy, x_start + dx:x_end + dx] = \
                        result[y_start:y_end, x_start:x_end]
                    eroded &= shifted
        result = eroded
    return result

def extract_contours_marching_squares(grid):
    """
    Extraction de contours via Marching Squares simplifié.
    Returns: liste de contours, chaque contour est une liste de points (y, x)
    """
    h, w = grid.shape
    contours = []
    visited = np.zeros_like(grid, dtype=bool)

    # Directions: droite, bas, gauche, haut
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    for y in range(h):
        for x in range(w):
            if grid[y, x] and not visited[y, x]:
                # Nouveau contour trouvé
                contour = []
                stack = [(y, x)]
                temp_visited = set()

                while stack:
                    cy, cx = stack.pop()
                    if (cy, cx) in temp_visited:
                        continue
                    temp_visited.add((cy, cx))

                    # Vérifier si c'est un point de bordure
                    is_border = False
                    for dy, dx in dirs:
                        ny, nx = cy + dy, cx + dx
                        if ny < 0 or ny >= h or nx < 0 or nx >= w or not grid[ny, nx]:
                            is_border = True
                            break

                    if is_border:
                        contour.append((cy, cx))

                    # Ajouter voisins
                    for dy, dx in dirs:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and grid[ny, nx] and (ny, nx) not in temp_visited:
                            stack.append((ny, nx))

                for cy, cx in temp_visited:
                    visited[cy, cx] = True

                if len(contour) >= 4:
                    contours.append(contour)

    return contours

def ramer_douglas_peucker(points, epsilon=1.0):
    """
    Algorithme de simplification Ramer-Douglas-Peucker.
    Args:
        points: liste de (x, y)
        epsilon: distance maximale autorisée
    Returns:
        points simplifiés
    """
    if len(points) < 3:
        return points

    # Trouver le point le plus éloigné de la ligne start-end
    start = np.array(points[0])
    end = np.array(points[-1])

    max_dist = 0
    max_idx = 0

    for i in range(1, len(points) - 1):
        point = np.array(points[i])
        # Distance point-à-ligne
        line_vec = end - start
        line_len = np.linalg.norm(line_vec)
        if line_len == 0:
            dist = np.linalg.norm(point - start)
        else:
            line_unitvec = line_vec / line_len
            point_vec = point - start
            proj_length = np.dot(point_vec, line_unitvec)
            proj = start + proj_length * line_unitvec
            dist = np.linalg.norm(point - proj)

        if dist > max_dist:
            max_dist = dist
            max_idx = i

    # Si le point le plus éloigné dépasse epsilon, subdiviser
    if max_dist > epsilon:
        left = ramer_douglas_peucker(points[:max_idx + 1], epsilon)
        right = ramer_douglas_peucker(points[max_idx:], epsilon)
        return left[:-1] + right
    else:
        return [points[0], points[-1]]

def order_contour_points(points):
    """Ordonne les points d'un contour dans le sens trigonométrique"""
    if len(points) < 3:
        return points

    # Calculer le centroïde
    points_arr = np.array(points)
    centroid = points_arr.mean(axis=0)

    # Calculer les angles
    angles = np.arctan2(points_arr[:, 1] - centroid[1], points_arr[:, 0] - centroid[0])

    # Trier par angle
    sorted_indices = np.argsort(angles)
    return points_arr[sorted_indices].tolist()

# ==============================================================================
# PROPRIETES - ORDRE IMPORTANT: LidarItem AVANT LidarProProperties
# ==============================================================================
class LidarItem(bpy.types.PropertyGroup):
    """Item de la liste des nuages de points"""
    obj: bpy.props.PointerProperty(
        name="Objet Lidar",
        type=bpy.types.Object,
        description="Référence vers l'objet nuage de points"
    )

class LidarProProperties(bpy.types.PropertyGroup):
    """Propriétés principales du plugin"""
    lidar_list: bpy.props.CollectionProperty(type=LidarItem)
    active_lidar_index: bpy.props.IntProperty(default=0)

    # Import
    import_decimation: bpy.props.IntProperty(
        name="Décimation Sol",
        description="Conserver 1 point sur N pour la classe Sol (2)",
        default=1,
        min=1,
        max=100
    )
    import_class_ground: bpy.props.BoolProperty(
        name="Sol (Classe 2)",
        description="Importer les points de classe 2 (Sol)",
        default=True
    )
    import_class_veg: bpy.props.BoolProperty(
        name="Végétation (3,4,5)",
        description="Importer les points de classes 3, 4, 5 (Végétation)",
        default=True
    )
    import_class_build: bpy.props.BoolProperty(
        name="Bâtiments (Classe 6)",
        description="Importer les points de classe 6 (Bâtiments)",
        default=True
    )

    master_offset_x: bpy.props.FloatProperty()
    master_offset_y: bpy.props.FloatProperty()
    master_offset_z: bpy.props.FloatProperty()
    has_master_offset: bpy.props.BoolProperty(default=False)

    # Terrain
    terrain_precision: bpy.props.FloatProperty(
        name="Résolution",
        description="Espacement de la grille de terrain en mètres",
        default=1.0,
        min=0.1,
        max=10.0
    )
    chunk_size: bpy.props.IntProperty(
        name="Taille Chunk",
        description="Taille des morceaux de traitement en mètres",
        default=50,
        min=10,
        max=200
    )
    apply_material: bpy.props.BoolProperty(
        name="Appliquer Matériaux",
        description="Appliquer automatiquement les matériaux",
        default=True
    )
    texture_resolution: bpy.props.FloatProperty(
        name="Résolution Texture",
        description="Mètres par pixel de texture",
        default=1.0,
        min=0.1,
        max=5.0
    )
    merge_threshold: bpy.props.FloatProperty(
        name="Distance Fusion",
        description="Distance maximale pour fusionner les vertices",
        default=0.15,
        min=0.01,
        max=1.0
    )
    terrain_fill_gaps: bpy.props.BoolProperty(
        name="Combler les Trous",
        description="Élargir les zones de terrain pour éviter les trous",
        default=True
    )
    terrain_overlap: bpy.props.FloatProperty(
        name="Chevauchement",
        description="Distance de chevauchement entre chunks (en mètres)",
        default=5.0,
        min=0.0,
        max=20.0
    )

    # Végétation
    veg_density_factor: bpy.props.FloatProperty(
        name="Densité",
        description="Pourcentage d'arbres à conserver",
        default=0.8,
        min=0.01,
        max=1.0,
        subtype='PERCENTAGE'
    )
    veg_grid_size: bpy.props.FloatProperty(
        name="Grille",
        description="Taille de la grille de clustering en mètres",
        default=2.0,
        min=0.5,
        max=10.0
    )
    min_points_veg: bpy.props.IntProperty(
        name="Points Min",
        description="Nombre minimum de points pour créer un arbre",
        default=8,
        min=3,
        max=50
    )
    height_bush_max: bpy.props.FloatProperty(
        name="Hauteur Buisson Max",
        description="Hauteur maximale pour considérer comme buisson",
        default=2.5,
        min=0.5,
        max=5.0
    )
    analyze_radius: bpy.props.FloatProperty(
        name="Rayon Analyse",
        description="Rayon d'analyse pour détecter les conifères",
        default=2.5,
        min=0.5,
        max=10.0
    )
    density_threshold: bpy.props.IntProperty(
        name="Seuil Conifère",
        description="Nombre de points pour considérer comme conifère",
        default=150,
        min=50,
        max=500
    )
    scale_variation: bpy.props.FloatProperty(
        name="Variation Taille",
        description="Variation aléatoire de la taille des arbres",
        default=0.2,
        min=0.0,
        max=0.5
    )
    rotation_random: bpy.props.BoolProperty(
        name="Rotation Aléatoire",
        description="Appliquer une rotation Z aléatoire",
        default=True
    )
    veg_use_geonodes: bpy.props.BoolProperty(
        name="Utiliser Geometry Nodes",
        description="Instancier avec Geometry Nodes (plus performant)",
        default=True
    )

    # Bâtiments (VECTORIEL GEOMETRIQUE)
    build_alpha_shape: bpy.props.FloatProperty(
        name="Alpha Shape",
        description="Paramètre de concavité du contour (plus petit = plus précis)",
        default=3.0,
        min=0.5,
        max=10.0
    )
    build_dbscan_eps: bpy.props.FloatProperty(
        name="DBSCAN Epsilon",
        description="Distance maximale entre points d'un même cluster (m)",
        default=1.5,
        min=0.5,
        max=5.0
    )
    min_build_points: bpy.props.IntProperty(
        name="Points Min",
        description="Nombre minimum de points pour créer un bâtiment",
        default=15,
        min=5,
        max=100
    )
    min_build_area: bpy.props.FloatProperty(
        name="Surface Min",
        description="Surface minimale d'un bâtiment (m²)",
        default=20.0,
        min=5.0,
        max=100.0
    )
    build_limit_enabled: bpy.props.BoolProperty(
        name="Limiter Nombre",
        description="Activer la limitation du nombre de bâtiments (pour tests)",
        default=False
    )
    build_limit_count: bpy.props.IntProperty(
        name="Max Bâtiments",
        description="Nombre maximum de bâtiments à générer (0 = illimité)",
        default=5,
        min=1,
        max=100
    )
    build_apply_materials: bpy.props.BoolProperty(
        name="Matériaux Procéduraux",
        description="Générer des matériaux procéduraux pour les bâtiments",
        default=True
    )
    intensity_threshold_roof: bpy.props.FloatProperty(
        name="Seuil Intensité Toit",
        description="Valeur d'intensité séparant tuiles (foncé) et bac acier (clair)",
        default=0.5,
        min=0.0,
        max=1.0
    )

# ==============================================================================
# OPERATEURS
# ==============================================================================
class LIDARPRO_OT_AddToList(bpy.types.Operator):
    bl_idname = "lidarpro.add_to_list"
    bl_label = "Ajouter Sélection"
    bl_description = "Ajouter les objets sélectionnés à la liste de traitement"

    def execute(self, context):
        props = context.scene.lidar_pro
        added_count = 0

        for obj in context.selected_objects:
            if obj.type == 'MESH':
                exists = any(item.obj == obj for item in props.lidar_list)

                if not exists:
                    item = props.lidar_list.add()
                    item.obj = obj
                    obj.hide_viewport = True
                    added_count += 1

        if added_count > 0:
            self.report({'INFO'}, f"{added_count} objet(s) ajouté(s)")
        else:
            self.report({'WARNING'}, "Aucun nouvel objet ajouté")

        return {'FINISHED'}

class LIDARPRO_OT_ClearList(bpy.types.Operator):
    bl_idname = "lidarpro.clear_list"
    bl_label = "Vider Liste"
    bl_description = "Vider complètement la liste de traitement"

    def execute(self, context):
        props = context.scene.lidar_pro

        for item in props.lidar_list:
            if item.obj:
                item.obj.hide_viewport = False

        props.lidar_list.clear()
        self.report({'INFO'}, "Liste vidée")
        return {'FINISHED'}

class LIDARPRO_OT_RemoveFromList(bpy.types.Operator):
    bl_idname = "lidarpro.remove_from_list"
    bl_label = "Retirer"
    bl_description = "Retirer l'objet sélectionné de la liste"

    def execute(self, context):
        props = context.scene.lidar_pro
        idx = props.active_lidar_index

        if 0 <= idx < len(props.lidar_list):
            item = props.lidar_list[idx]
            if item.obj:
                item.obj.hide_viewport = False
            props.lidar_list.remove(idx)
            self.report({'INFO'}, "Objet retiré de la liste")

        return {'FINISHED'}

class LIDARPRO_OT_ToggleVisibility(bpy.types.Operator):
    bl_idname = "lidarpro.toggle_visibility"
    bl_label = "Tout Masquer/Afficher"
    bl_description = "Basculer la visibilité de tous les objets de la liste"

    def execute(self, context):
        props = context.scene.lidar_pro

        visible_count = sum(1 for item in props.lidar_list if item.obj and not item.obj.hide_viewport)
        should_hide = visible_count > len(props.lidar_list) / 2

        for item in props.lidar_list:
            if item.obj:
                item.obj.hide_viewport = should_hide

        return {'FINISHED'}

class LIDARPRO_OT_OpenIGNMap(bpy.types.Operator):
    bl_idname = "lidarpro.open_ign_map"
    bl_label = "📍 Télécharger Données IGN"
    bl_description = "Ouvrir le site de téléchargement des données LiDAR HD de l'IGN"

    def execute(self, context):
        webbrowser.open("https://cartes.gouv.fr/telechargement/IGNF_NUAGES-DE-POINTS-LIDAR-HD")
        return {'FINISHED'}

class LIDARPRO_OT_InstallLaspy(bpy.types.Operator):
    bl_idname = "lidarpro.install_laspy"
    bl_label = "Installer Laspy"
    bl_description = "Tenter d'installer automatiquement la bibliothèque laspy"

    def execute(self, context):
        if install_laspy():
            self.report({'INFO'}, "Laspy installé ! Redémarrez Blender")
        else:
            self.report({'ERROR'}, "Échec installation automatique")
        return {'FINISHED'}

class LIDARPRO_OT_ResetOffset(bpy.types.Operator):
    bl_idname = "lidarpro.reset_offset"
    bl_label = "Réinitialiser Position"
    bl_description = "Réinitialiser l'offset de position maître"

    def execute(self, context):
        props = context.scene.lidar_pro
        props.has_master_offset = False
        props.master_offset_x = 0
        props.master_offset_y = 0
        props.master_offset_z = 0
        self.report({'INFO'}, "Offset réinitialisé")
        return {'FINISHED'}

class LIDARPRO_OT_ImportLas(bpy.types.Operator, ImportHelper):
    bl_idname = "lidarpro.import_las"
    bl_label = "Importer Fichiers LAS/LAZ"
    bl_description = "Importer un ou plusieurs fichiers LAS/LAZ"
    bl_options = {'PRESET', 'UNDO'}

    files: bpy.props.CollectionProperty(type=bpy.types.OperatorFileListElement)
    directory: bpy.props.StringProperty(subtype='DIR_PATH')
    filter_glob: bpy.props.StringProperty(default="*.las;*.laz", options={'HIDDEN'})

    def execute(self, context):
        props = context.scene.lidar_pro
        import laspy

        file_paths = [os.path.join(self.directory, f.name) for f in self.files]
        if not file_paths:
            file_paths = [self.filepath]

        wm = context.window_manager
        wm.progress_begin(0, len(file_paths))

        print(f"\n{'='*60}")
        print(f"IMPORT DE {len(file_paths)} FICHIER(S) LAS/LAZ")
        print(f"{'='*60}")

        for i, filepath in enumerate(file_paths):
            wm.progress_update(i)

            for area in context.screen.areas:
                area.tag_redraw()

            print(f"[{i+1}/{len(file_paths)}] Import: {os.path.basename(filepath)}")

            try:
                las = laspy.read(filepath)
                x, y, z = np.array(las.x), np.array(las.y), np.array(las.z)

                try:
                    classes = np.array(las.classification, dtype=np.int32)
                except:
                    classes = np.zeros(len(x), dtype=np.int32)

                try:
                    intensity = np.array(las.intensity, dtype=np.float32)
                except:
                    intensity = None

                # Filtrage par classe
                keep_mask = np.zeros(len(x), dtype=bool)
                if props.import_class_ground:
                    keep_mask |= (classes == 2)
                if props.import_class_veg:
                    keep_mask |= np.isin(classes, [3, 4, 5])
                if props.import_class_build:
                    keep_mask |= (classes == 6)

                # Décimation du sol
                if props.import_decimation > 1:
                    deci_mask = np.zeros(len(x), dtype=bool)
                    deci_mask[::props.import_decimation] = True
                    keep_mask &= (~(classes == 2) | ((classes == 2) & deci_mask))

                x, y, z = x[keep_mask], y[keep_mask], z[keep_mask]
                classes = classes[keep_mask]
                if intensity is not None:
                    intensity = intensity[keep_mask]

                if len(x) == 0:
                    continue

                # Offset maître
                if not props.has_master_offset:
                    props.master_offset_x = float(np.min(x))
                    props.master_offset_y = float(np.min(y))
                    props.master_offset_z = float(np.min(z))
                    props.has_master_offset = True

                coords = np.vstack((
                    x - props.master_offset_x,
                    y - props.master_offset_y,
                    z - props.master_offset_z
                )).transpose()

                # Création mesh (OPTIMISE: from_pydata au lieu de bmesh)
                mesh = bpy.data.meshes.new(name=f"Lidar_{os.path.basename(filepath)}")
                mesh.from_pydata(coords, [], [])

                # Attributs
                mesh.attributes.new(name="scalar_Classification", type='INT', domain='POINT').data.foreach_set('value', classes)
                if intensity is not None:
                    mesh.attributes.new(name="scalar_Intensity", type='FLOAT', domain='POINT').data.foreach_set('value', intensity)

                # Création objet
                obj = bpy.data.objects.new(f"PC_{os.path.basename(filepath)}", mesh)
                context.scene.collection.objects.link(obj)
                obj["lidar_offset"] = [props.master_offset_x, props.master_offset_y, props.master_offset_z]

                # Ajout à la liste
                item = props.lidar_list.add()
                item.obj = obj
                obj.hide_viewport = True

            except Exception as e:
                print(f"Erreur lors de l'import de {filepath}: {e}")
                self.report({'ERROR'}, f"Erreur: {os.path.basename(filepath)}")

        wm.progress_end()

        print(f"{'='*60}")
        print(f"✓ IMPORT TERMINE: {len(file_paths)} fichier(s)")
        print(f"{'='*60}\n")

        self.report({'INFO'}, f"{len(file_paths)} fichier(s) importé(s)")
        return {'FINISHED'}

class LIDARPRO_OT_GenerateTerrain(bpy.types.Operator):
    bl_idname = "lidarpro.generate_terrain"
    bl_label = "Générer Terrain"
    bl_description = "Générer le terrain à partir des nuages de points de la liste"

    def execute(self, context):
        props = context.scene.lidar_pro
        lidar_items = [item for item in props.lidar_list if item.obj]

        if not lidar_items:
            self.report({'WARNING'}, "Aucun objet dans la liste")
            return {'CANCELLED'}

        mat = AssetLibraryManager.link_material("mat_terrain")
        merged_obj = bpy.data.objects.get("Terrain_Lidar_Merged")
        temp_terrains = []

        wm = context.window_manager
        total_steps = len(lidar_items) + 2
        wm.progress_begin(0, total_steps)

        print(f"\n{'='*60}")
        print(f"GENERATION TERRAIN - {len(lidar_items)} dalle(s)")
        print(f"{'='*60}")

        for i, item in enumerate(lidar_items):
            wm.progress_update(i)

            for area in context.screen.areas:
                area.tag_redraw()

            print(f"[{i+1}/{len(lidar_items)}] Traitement: {item.obj.name}")

            t_obj = self.process_terrain(context, item.obj, props, mat)
            if t_obj:
                temp_terrains.append(t_obj)

        if not temp_terrains:
            wm.progress_end()
            self.report({'WARNING'}, "Aucun terrain généré")
            return {'FINISHED'}

        # Fusion
        wm.progress_update(len(lidar_items))
        for area in context.screen.areas:
            area.tag_redraw()

        print(f"\n[FUSION] Assemblage de {len(temp_terrains)} dalle(s)...")

        bpy.ops.object.select_all(action='DESELECT')
        if merged_obj:
            merged_obj.select_set(True)
            context.view_layer.objects.active = merged_obj
        else:
            temp_terrains[0].select_set(True)
            context.view_layer.objects.active = temp_terrains[0]

        for t in temp_terrains:
            t.select_set(True)

        bpy.ops.object.join()

        final = context.active_object
        final.name = "Terrain_Lidar_Merged"

        # Nettoyage
        wm.progress_update(len(lidar_items) + 1)
        for area in context.screen.areas:
            area.tag_redraw()

        print(f"[NETTOYAGE] Suppression des doublons...")

        bm = bmesh.new()
        bm.from_mesh(final.data)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=props.merge_threshold)
        bm.to_mesh(final.data)
        bm.free()

        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.delete_loose()
        bpy.ops.object.mode_set(mode='OBJECT')

        wm.progress_end()

        print(f"{'='*60}")
        print(f"✓ TERRAIN GENERE: {len(temp_terrains)} dalle(s) fusionnée(s)")
        print(f"{'='*60}\n")

        self.report({'INFO'}, f"Terrain généré avec {len(temp_terrains)} dalle(s)")
        return {'FINISHED'}

    def process_terrain(self, context, lidar, props, t_mat):
        """OPTIMISE: Opérations vectorielles NumPy pures"""
        mesh = lidar.data
        n = len(mesh.vertices)
        pos = np.zeros(n*3, dtype=np.float32)
        mesh.vertices.foreach_get('co', pos)
        pos = pos.reshape((-1,3))

        mask = np.ones(n, dtype=bool)
        if 'scalar_Classification' in mesh.attributes:
            cls = np.zeros(n, dtype=np.int32)
            mesh.attributes['scalar_Classification'].data.foreach_get('value', cls)
            mask = (cls == 2)
            if np.sum(mask) == 0:
                mask[:] = True

        inte = None
        if 'scalar_Intensity' in mesh.attributes:
            inte = np.zeros(n, dtype=np.float32)
            mesh.attributes['scalar_Intensity'].data.foreach_get('value', inte)
            inte = inte[mask]

        mat_w = np.array(lidar.matrix_world)
        pts = pos[mask] @ mat_w[:3,:3].T + mat_w[:3,3]

        if len(pts) == 0:
            return None

        min_x, min_y = pts[:,0].min(), pts[:,1].min()
        w, h = pts[:,0].max() - min_x, pts[:,1].max() - min_y

        # Texture
        img = None
        if inte is not None and props.apply_material:
            tw, th = int(np.ceil(w/props.texture_resolution)), int(np.ceil(h/props.texture_resolution))
            if tw > 0 and th > 0:
                imin, imax = inte.min(), inte.max()
                n_int = (inte - imin) / (imax - imin) if imax > imin else inte
                gx = ((pts[:,0]-min_x)/w*(tw-1)).astype(np.int32)
                gy = ((pts[:,1]-min_y)/h*(th-1)).astype(np.int32)
                g_sum = np.zeros((th,tw), dtype=np.float32)
                g_cnt = np.zeros((th,tw), dtype=np.float32)
                idx = gy * tw + gx
                np.add.at(g_sum.ravel(), idx, n_int)
                np.add.at(g_cnt.ravel(), idx, 1)
                valid = g_cnt > 0
                final = np.zeros_like(g_sum)
                final[valid] = g_sum[valid]/g_cnt[valid]
                px = np.dstack((final, final, final, np.ones_like(final))).flatten()
                name = f"Tex_{lidar.name}"
                if name in bpy.data.images:
                    bpy.data.images.remove(bpy.data.images[name])
                img = bpy.data.images.new(name, width=tw, height=th)
                img.pixels.foreach_set(px)
                img.pack()

        # Triangulation optimisée
        final_verts, final_faces, v_off = [], [], 0
        grid = np.round(pts[:,:2]/props.terrain_precision)*props.terrain_precision
        _, u_idx = np.unique(np.ascontiguousarray(grid).view([('x',float),('y',float)]), return_index=True)
        pts_ds = pts[u_idx]

        overlap = props.terrain_overlap if props.terrain_fill_gaps else 0
        buffer = max(10, overlap)

        for xs in np.arange(min_x, min_x+w, props.chunk_size):
            for ys in np.arange(min_y, min_y+h, props.chunk_size):
                chunk = pts_ds[
                    (pts_ds[:,0] >= xs - buffer) &
                    (pts_ds[:,0] <= xs + props.chunk_size + buffer) &
                    (pts_ds[:,1] >= ys - buffer) &
                    (pts_ds[:,1] <= ys + props.chunk_size + buffer)
                ]

                if len(chunk) < 3:
                    continue

                try:
                    res = geometry.delaunay_2d_cdt([Vector(p[:2]) for p in chunk], [], [], 0, 0.001)
                    cv = np.array([(v.x, v.y, chunk[res[3][i][0]][2] if res[3][i] else 0) for i,v in enumerate(res[0])])
                    cf = np.array(res[2])
                    ct = (cv[cf[:,0]]+cv[cf[:,1]]+cv[cf[:,2]])/3.0

                    mask_f = (
                        (ct[:,0] >= xs - overlap) &
                        (ct[:,0] < xs + props.chunk_size + overlap) &
                        (ct[:,1] >= ys - overlap) &
                        (ct[:,1] < ys + props.chunk_size + overlap)
                    )

                    if np.any(mask_f):
                        final_verts.extend(cv)
                        final_faces.extend(cf[mask_f] + v_off)
                        v_off += len(cv)
                except:
                    pass

        if final_verts:
            me = bpy.data.meshes.new(f"Terrain_Temp")
            me.from_pydata(final_verts, [], final_faces)
            obj = bpy.data.objects.new(me.name, me)
            col = bpy.data.collections.get("Lidar_Terrain") or bpy.data.collections.new("Lidar_Terrain")
            if "Lidar_Terrain" not in context.scene.collection.children:
                context.scene.collection.children.link(col)
            col.objects.link(obj)

            # UVs
            uv = me.uv_layers.new(name="UVMap").data
            vco = np.zeros(len(me.vertices)*3, dtype=np.float32)
            me.vertices.foreach_get("co", vco)
            vco = vco.reshape((-1,3))
            u, v = (vco[:,0]-min_x)/w, (vco[:,1]-min_y)/h
            l_idx = np.zeros(len(me.loops), dtype=np.int32)
            me.loops.foreach_get("vertex_index", l_idx)
            uv.foreach_set("uv", np.column_stack((u[l_idx], v[l_idx])).flatten())

            # Matériau
            if t_mat:
                m = t_mat.copy()
                obj.data.materials.append(m)
                if img and m.use_nodes:
                    tn = next((n for n in m.node_tree.nodes if n.type=='TEX_IMAGE'), None)
                    if not tn:
                        bsdf = m.node_tree.nodes.get("Principled BSDF")
                        if bsdf:
                            tn = m.node_tree.nodes.new('ShaderNodeTexImage')
                            m.node_tree.links.new(tn.outputs['Color'], bsdf.inputs['Base Color'])
                    if tn:
                        tn.image = img
            return obj
        return None

class LIDARPRO_OT_GenerateVegetation(bpy.types.Operator):
    bl_idname = "lidarpro.generate_vegetation"
    bl_label = "Générer Végétation"
    bl_description = "Générer la végétation à partir des nuages de points de la liste"

    def execute(self, context):
        props = context.scene.lidar_pro
        lidar_items = [item for item in props.lidar_list if item.obj]
        terr = bpy.data.objects.get("Terrain_Lidar_Merged")

        if not terr:
            self.report({'WARNING'}, "Terrain introuvable (générez-le d'abord)")
            return {'CANCELLED'}

        print(f"\n{'='*60}")
        print(f"GENERATION VEGETATION - {len(lidar_items)} dalle(s)")
        print(f"Mode: {'Geometry Nodes' if props.veg_use_geonodes else 'Objets individuels'}")
        print(f"{'='*60}")

        if props.veg_use_geonodes:
            return self.execute_geonodes_mode(context, lidar_items, terr, props)
        else:
            return self.execute_legacy_mode(context, lidar_items, terr, props)

    def execute_geonodes_mode(self, context, lidar_items, terr, props):
        """
        NOUVEAU: Mode Geometry Nodes
        Crée UN SEUL mesh de points, puis ajoute un modificateur Geometry Nodes
        pour instancier la collection dessus.
        """
        # Chargement des assets
        LIB = ["1_bush", "2_bush", "3_conifere", "4_conifere", "5_conifere", "6_feuillu", "7_feuillu"]
        MDATA = {}

        for n in LIB:
            AssetLibraryManager.link_object(n, False)
            o = bpy.data.objects.get(n)
            if o:
                bb = [Vector(b) for b in o.bound_box]
                zs = [b.z for b in bb]
                MDATA[n] = {'o': o, 'h': (max(zs)-min(zs))*o.scale.z, 'z': min(zs)}

        # Collection de végétation
        col = bpy.data.collections.get("Vegetation_Generated") or bpy.data.collections.new("Vegetation_Generated")
        if "Vegetation_Generated" not in context.scene.collection.children:
            context.scene.collection.children.link(col)

        # Préparer les assets dans la collection pour Geometry Nodes
        assets_col = bpy.data.collections.get("Vegetation_Assets") or bpy.data.collections.new("Vegetation_Assets")
        if "Vegetation_Assets" not in context.scene.collection.children:
            context.scene.collection.children.link(assets_col)

        for n in LIB:
            o = bpy.data.objects.get(n)
            if o and o.name not in assets_col.objects:
                assets_col.objects.link(o)

        dg = context.evaluated_depsgraph_get()
        tbvh = BVHTree.FromObject(terr.evaluated_get(dg), dg)

        wm = context.window_manager
        wm.progress_begin(0, len(lidar_items))

        # Accumuler tous les points de végétation
        all_points = []
        all_types = []  # Pour stocker le type d'arbre (0=bush, 1=conifère, 2=feuillu)
        all_scales = []
        all_rotations = []

        for idx_l, item in enumerate(lidar_items):
            wm.progress_update(idx_l)

            for area in context.screen.areas:
                area.tag_redraw()

            lidar = item.obj
            print(f"[{idx_l+1}/{len(lidar_items)}] Traitement: {lidar.name}")

            mesh = lidar.data
            n = len(mesh.vertices)

            co = np.zeros(n*3, dtype=np.float32)
            mesh.vertices.foreach_get('co', co)
            co = co.reshape((-1,3))

            if 'scalar_Classification' not in mesh.attributes:
                continue

            cls = np.zeros(n, dtype=np.int32)
            mesh.attributes['scalar_Classification'].data.foreach_get('value', cls)
            mask = np.isin(cls, [3,4,5])

            pts_loc = co[mask]
            pts_cls = cls[mask]

            mw = np.array(lidar.matrix_world)
            pts_w = pts_loc @ mw[:3,:3].T + mw[:3,3]

            if len(pts_w) == 0:
                continue

            # OPTIMISE: Clustering vectoriel par grille
            grid = np.floor(pts_w[:,:2]/props.veg_grid_size).astype(np.int64)
            hsh = grid[:,0]*1000000 + grid[:,1]
            srt = np.argsort(hsh)
            pts_s, cls_s, hsh_s = pts_w[srt], pts_cls[srt], hsh[srt]
            _, sp = np.unique(hsh_s, return_index=True)
            c_pts, c_cls = np.split(pts_s, sp[1:]), np.split(cls_s, sp[1:])

            for i, p in enumerate(c_pts):
                if len(p) < props.min_points_veg or random.random() > props.veg_density_factor:
                    continue

                mode = np.bincount(c_cls[i]).argmax()
                zmin, zmax = p[:,2].min(), p[:,2].max()
                h = zmax - zmin
                cx, cy = np.mean(p[:,0]), np.mean(p[:,1])

                # Déterminer le type
                tree_type = 0  # bush par défaut
                if mode == 3 or h < props.height_bush_max:
                    tree_type = 0  # bush
                else:
                    top = np.argmax(p[:,2])
                    dens = np.sum(np.sum((p[:,:2]-p[top][:2])**2, axis=1) < props.analyze_radius**2)
                    if dens > props.density_threshold:
                        tree_type = 1  # conifère
                    else:
                        tree_type = 2  # feuillu

                # Ray cast pour trouver le sol
                l, _, _, _ = tbvh.ray_cast(Vector((cx,cy,zmax+100)), Vector((0,0,-1)))
                ground_z = l.z if l else zmin

                # Calculer échelle et rotation
                if tree_type == 0:  # bush
                    ref_h = MDATA.get("1_bush", {}).get('h', 1.0)
                    ts = max(1.5, min((h/ref_h if ref_h > 0 else 1.0) * 1.5, 4.0))
                else:
                    ref_h = MDATA.get("6_feuillu", {}).get('h', 5.0)
                    ts = max(0.2, min(h/ref_h if ref_h > 0 else 1.0, 2.5))

                s = ts * random.uniform(1.0-props.scale_variation, 1.0+props.scale_variation)
                rot = random.uniform(0, 6.28) if props.rotation_random else 0

                all_points.append((cx, cy, ground_z))
                all_types.append(tree_type)
                all_scales.append(s)
                all_rotations.append(rot)

        wm.progress_end()

        if len(all_points) == 0:
            print("Aucun point de végétation trouvé")
            self.report({'WARNING'}, "Aucun point de végétation")
            return {'FINISHED'}

        print(f"\n[GEOMETRY NODES] Création du mesh de points ({len(all_points)} points)...")

        # Créer le mesh de points
        veg_mesh = bpy.data.meshes.new("Vegetation_Points")
        veg_mesh.from_pydata(all_points, [], [])

        # Ajouter les attributs personnalisés
        # Type d'arbre
        type_attr = veg_mesh.attributes.new(name="tree_type", type='INT', domain='POINT')
        type_attr.data.foreach_set('value', all_types)

        # Échelle
        scale_attr = veg_mesh.attributes.new(name="tree_scale", type='FLOAT', domain='POINT')
        scale_attr.data.foreach_set('value', all_scales)

        # Rotation
        rot_attr = veg_mesh.attributes.new(name="tree_rotation", type='FLOAT', domain='POINT')
        rot_attr.data.foreach_set('value', all_rotations)

        # Créer l'objet
        veg_obj = bpy.data.objects.new("Vegetation_GeoNodes", veg_mesh)
        col.objects.link(veg_obj)

        # Ajouter le modificateur Geometry Nodes
        print("[GEOMETRY NODES] Ajout du modificateur...")

        # Créer ou récupérer le node tree
        node_tree_name = "VegetationInstancer"
        if node_tree_name in bpy.data.node_groups:
            node_tree = bpy.data.node_groups[node_tree_name]
        else:
            node_tree = bpy.data.node_groups.new(node_tree_name, 'GeometryNodeTree')
            self.setup_geometry_nodes(node_tree, assets_col)

        # Ajouter le modificateur
        mod = veg_obj.modifiers.new(name="Vegetation_Instancer", type='NODES')
        mod.node_group = node_tree

        print(f"{'='*60}")
        print(f"✓ VEGETATION GENEREE: {len(all_points)} instances")
        print(f"  Mode: Geometry Nodes (performant)")
        print(f"{'='*60}\n")

        self.report({'INFO'}, f"Végétation générée: {len(all_points)} instances (Geometry Nodes)")
        return {'FINISHED'}

    def setup_geometry_nodes(self, node_tree, assets_col):
        """Configure le node tree Geometry Nodes pour instancier les arbres"""
        nodes = node_tree.nodes
        links = node_tree.links
        nodes.clear()

        # Input/Output
        group_in = nodes.new('NodeGroupInput')
        group_in.location = (-400, 0)
        group_out = nodes.new('NodeGroupOutput')
        group_out.location = (600, 0)

        # Instance on Points
        instance_node = nodes.new('GeometryNodeInstanceOnPoints')
        instance_node.location = (200, 0)

        # Collection Info (pour charger la collection d'assets)
        col_info = nodes.new('GeometryNodeCollectionInfo')
        col_info.location = (0, -200)
        col_info.inputs['Collection'].default_value = assets_col
        col_info.transform_space = 'ORIGINAL'

        # Connections
        links.new(group_in.outputs['Geometry'], instance_node.inputs['Points'])
        links.new(col_info.outputs['Geometry'], instance_node.inputs['Instance'])
        links.new(instance_node.outputs['Instances'], group_out.inputs['Geometry'])

        # Setup inputs/outputs
        if not node_tree.interface.items_tree:
            node_tree.interface.new_socket(name='Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
            node_tree.interface.new_socket(name='Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')

        print("  → Node tree Geometry Nodes créé")

    def execute_legacy_mode(self, context, lidar_items, terr, props):
        """Mode legacy: Crée des objets individuels (conservé pour compatibilité)"""
        # (Code original conservé)
        LIB = ["1_bush", "2_bush", "3_conifere", "4_conifere", "5_conifere", "6_feuillu", "7_feuillu"]
        MDATA = {}

        for n in LIB:
            AssetLibraryManager.link_object(n, False)
            o = bpy.data.objects.get(n)
            if o:
                bb = [Vector(b) for b in o.bound_box]
                zs = [b.z for b in bb]
                MDATA[n] = {'o': o, 'h': (max(zs)-min(zs))*o.scale.z, 'z': min(zs)}

        col = bpy.data.collections.get("Vegetation_Generated") or bpy.data.collections.new("Vegetation_Generated")
        if "Vegetation_Generated" not in context.scene.collection.children:
            context.scene.collection.children.link(col)

        dg = context.evaluated_depsgraph_get()
        tbvh = BVHTree.FromObject(terr.evaluated_get(dg), dg)

        wm = context.window_manager
        wm.progress_begin(0, len(lidar_items))

        total_vegetation = 0

        for idx_l, item in enumerate(lidar_items):
            wm.progress_update(idx_l)

            for area in context.screen.areas:
                area.tag_redraw()

            lidar = item.obj
            print(f"[{idx_l+1}/{len(lidar_items)}] Traitement: {lidar.name}")

            mesh = lidar.data
            n = len(mesh.vertices)

            co = np.zeros(n*3, dtype=np.float32)
            mesh.vertices.foreach_get('co', co)
            co = co.reshape((-1,3))

            if 'scalar_Classification' not in mesh.attributes:
                continue

            cls = np.zeros(n, dtype=np.int32)
            mesh.attributes['scalar_Classification'].data.foreach_get('value', cls)
            mask = np.isin(cls, [3,4,5])

            pts_loc = co[mask]
            pts_cls = cls[mask]

            mw = np.array(lidar.matrix_world)
            pts_w = pts_loc @ mw[:3,:3].T + mw[:3,3]

            if len(pts_w) == 0:
                continue

            grid = np.floor(pts_w[:,:2]/props.veg_grid_size).astype(np.int64)
            hsh = grid[:,0]*1000000 + grid[:,1]
            srt = np.argsort(hsh)
            pts_s, cls_s, hsh_s = pts_w[srt], pts_cls[srt], hsh[srt]
            _, sp = np.unique(hsh_s, return_index=True)
            c_pts, c_cls = np.split(pts_s, sp[1:]), np.split(cls_s, sp[1:])

            count_this_tile = 0

            for i, p in enumerate(c_pts):
                if len(p) < props.min_points_veg or random.random() > props.veg_density_factor:
                    continue

                mode = np.bincount(c_cls[i]).argmax()
                zmin, zmax = p[:,2].min(), p[:,2].max()
                h = zmax - zmin
                cx, cy = np.mean(p[:,0]), np.mean(p[:,1])

                mname = None
                if mode == 3 or h < props.height_bush_max:
                    pool = [x for x in ["1_bush", "2_bush"] if x in MDATA]
                    if pool:
                        mname = random.choice(pool)
                else:
                    top = np.argmax(p[:,2])
                    dens = np.sum(np.sum((p[:,:2]-p[top][:2])**2, axis=1) < props.analyze_radius**2)
                    pool = [x for x in (["3_conifere", "4_conifere", "5_conifere"] if dens > props.density_threshold else ["6_feuillu", "7_feuillu"]) if x in MDATA]
                    if pool:
                        mname = random.choice(pool)

                if mname:
                    d = MDATA[mname]
                    l, _, _, _ = tbvh.ray_cast(Vector((cx,cy,zmax+100)), Vector((0,0,-1)))
                    ground_z = l.z if l else zmin

                    new = d['o'].copy()
                    new.data = d['o'].data

                    ts = (h/d['h'] if d['h']>0 else 1.0)
                    if "bush" in mname:
                        ts = max(1.5, min(ts * 1.5, 4.0))
                    else:
                        ts = max(0.2, min(ts, 2.5))

                    s = ts * random.uniform(1.0-props.scale_variation, 1.0+props.scale_variation)
                    new.scale = (s,s,s)
                    new.rotation_euler = (0,0, random.uniform(0, 6.28) if props.rotation_random else 0)
                    new.location = (cx, cy, ground_z - d['z']*s)
                    col.objects.link(new)
                    count_this_tile += 1

            total_vegetation += count_this_tile
            print(f"  → {count_this_tile} éléments de végétation créés")

        wm.progress_end()

        print(f"{'='*60}")
        print(f"✓ VEGETATION GENEREE: {total_vegetation} éléments")
        print(f"{'='*60}\n")

        self.report({'INFO'}, f"Végétation générée: {total_vegetation} éléments")
        return {'FINISHED'}

class LIDARPRO_OT_GenerateBuildings(bpy.types.Operator):
    bl_idname = "lidarpro.generate_buildings"
    bl_label = "Générer Bâtiments"
    bl_description = "Générer les bâtiments avec l'algorithme VECTORIEL GEOMETRIQUE"

    def execute(self, context):
        props = context.scene.lidar_pro
        lidar_items = [item for item in props.lidar_list if item.obj]
        terr = bpy.data.objects.get("Terrain_Lidar_Merged")

        if not terr:
            self.report({'WARNING'}, "Terrain introuvable (générez-le d'abord)")
            return {'CANCELLED'}

        dg = context.evaluated_depsgraph_get()
        tbvh = BVHTree.FromObject(terr.evaluated_get(dg), dg)

        col = bpy.data.collections.get("Buildings_Generated") or bpy.data.collections.new("Buildings_Generated")
        if "Buildings_Generated" not in context.scene.collection.children:
            context.scene.collection.children.link(col)

        mr_default = AssetLibraryManager.link_material("Mat_Roof") or bpy.data.materials.new("Mat_Roof_Default")
        mw_default = AssetLibraryManager.link_material("Mat_Wall") or bpy.data.materials.new("Mat_Wall_Default")

        wm = context.window_manager
        wm.progress_begin(0, len(lidar_items))

        print(f"\n{'='*60}")
        print(f"GENERATION BATIMENTS V17 - VECTORIEL GEOMETRIQUE")
        print(f"Alpha Shape: {props.build_alpha_shape}m")
        print(f"DBSCAN eps: {props.build_dbscan_eps}m")
        print(f"Min points: {props.min_build_points}")
        if props.build_limit_enabled:
            print(f"⚠ MODE TEST: Limité à {props.build_limit_count} bâtiment(s)")
        print(f"{'='*60}")

        total_buildings = 0

        for idx_l, item in enumerate(lidar_items):
            wm.progress_update(idx_l)

            for area in context.screen.areas:
                area.tag_redraw()

            lidar = item.obj
            print(f"\n[{idx_l+1}/{len(lidar_items)}] Traitement: {lidar.name}")

            mesh = lidar.data
            n = len(mesh.vertices)
            pos = np.zeros(n*3, dtype=np.float32)
            mesh.vertices.foreach_get('co', pos)
            pos = pos.reshape((-1,3))

            if 'scalar_Classification' not in mesh.attributes:
                print("  ⚠ Pas d'attribut de classification")
                continue

            cls = np.zeros(n, dtype=np.int32)
            mesh.attributes['scalar_Classification'].data.foreach_get('value', cls)
            mask = (cls == 6)

            intensity_data = None
            if 'scalar_Intensity' in mesh.attributes:
                intensity_data = np.zeros(n, dtype=np.float32)
                mesh.attributes['scalar_Intensity'].data.foreach_get('value', intensity_data)
                intensity_data = intensity_data[mask]

            pts_w = pos[mask] @ np.array(lidar.matrix_world)[:3,:3].T + np.array(lidar.matrix_world)[:3,3]

            if len(pts_w) == 0:
                print("  ⚠ Aucun point de classe 6 (bâtiments)")
                continue

            print(f"  → {len(pts_w)} points de bâtiments détectés")

            # === NOUVEL ALGORITHME VECTORIEL GEOMETRIQUE ===

            # ÉTAPE A: Filtrage des façades
            print(f"  [A - FILTRAGE FACADES] Analyse de verticalité...")
            roof_mask = filter_facade_points(pts_w, k_neighbors=10, variance_threshold=3.0)
            roof_points = pts_w[roof_mask]

            if intensity_data is not None:
                roof_intensity = intensity_data[roof_mask]
            else:
                roof_intensity = None

            print(f"  → {len(roof_points)} points de toit conservés ({100*len(roof_points)/len(pts_w):.1f}%)")

            if len(roof_points) < props.min_build_points:
                print("  ⚠ Pas assez de points de toit")
                continue

            # ÉTAPE B: Clustering DBSCAN avec KDTree optimisé
            print(f"  [B - CLUSTERING] DBSCAN KDTree (eps={props.build_dbscan_eps}m)...")

            # Construire KDTree pour les points de toit (réutilisé par DBSCAN)
            from mathutils.kdtree import KDTree
            kd_roof = KDTree(len(roof_points))
            for i, pt in enumerate(roof_points):
                kd_roof.insert((pt[0], pt[1], 0.0), i)
            kd_roof.balance()

            labels = dbscan_kdtree(roof_points[:, :2], eps=props.build_dbscan_eps, min_samples=props.min_build_points, kd_tree=kd_roof)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            print(f"  → {n_clusters} clusters trouvés")

            if n_clusters == 0:
                print("  ⚠ Aucun cluster valide")
                continue

            # Traiter chaque cluster (= chaque bâtiment)
            count_this_tile = 0

            for cluster_id in range(n_clusters):
                # Vérifier la limite de bâtiments si activée
                if props.build_limit_enabled and total_buildings >= props.build_limit_count:
                    print(f"  ⚠ Limite de {props.build_limit_count} bâtiments atteinte - Arrêt")
                    break

                cluster_mask = (labels == cluster_id)
                cluster_points = roof_points[cluster_mask]

                if len(cluster_points) < props.min_build_points:
                    continue

                # ÉTAPE C: Contour Concave (Alpha Shape)
                print(f"  [C - ALPHA SHAPE] Cluster {cluster_id+1}/{n_clusters}...")
                contour_2d = alpha_shape_2d(cluster_points[:, :2], alpha=props.build_alpha_shape)

                if len(contour_2d) < 3:
                    print(f"    ⚠ Contour invalide")
                    continue

                # ÉTAPE D: Régularisation (angles droits)
                print(f"  [D - RECTIFICATION] Régularisation des angles...")
                contour_rectified = rectify_polygon_to_right_angles(contour_2d, angle_tolerance=15.0)

                if len(contour_rectified) < 3:
                    print(f"    ⚠ Rectification échouée")
                    continue

                # Calculer la surface
                contour_arr = np.array(contour_rectified)
                area = 0.5 * abs(np.sum(contour_arr[:-1, 0] * contour_arr[1:, 1] -
                                        contour_arr[1:, 0] * contour_arr[:-1, 1]))

                if area < props.min_build_area:
                    print(f"    ⚠ Surface trop petite ({area:.1f}m²)")
                    continue

                # Calculer hauteur du toit (médiane haute pour éviter outliers)
                roof_height = np.percentile(cluster_points[:, 2], 85)

                # Intensité moyenne du cluster
                intensity_avg = 0.5
                if roof_intensity is not None:
                    cluster_intensity = roof_intensity[cluster_mask]
                    if len(cluster_intensity) > 0:
                        i_min, i_max = cluster_intensity.min(), cluster_intensity.max()
                        if i_max > i_min:
                            intensity_avg = (cluster_intensity.mean() - i_min) / (i_max - i_min)

                # ÉTAPE E: Génération 3D
                print(f"  [E - GENERATION 3D] Création mesh...")
                obj = self.create_building_mesh_vectorial(
                    context, contour_rectified, roof_height, intensity_avg,
                    tbvh, props, mr_default, mw_default,
                    f"{lidar.name}_{cluster_id}"
                )

                if obj:
                    col.objects.link(obj)
                    count_this_tile += 1
                    total_buildings += 1
                    print(f"    ✓ Bâtiment créé ({area:.1f}m², h={roof_height:.1f}m)")

            if count_this_tile > 0:
                print(f"  ✓ {count_this_tile} bâtiments créés pour cette dalle")

            # Sortir de la boucle principale si limite atteinte
            if props.build_limit_enabled and total_buildings >= props.build_limit_count:
                print(f"\n  🛑 LIMITE ATTEINTE: {props.build_limit_count} bâtiments générés")
                break

        wm.progress_end()

        print(f"{'='*60}")
        print(f"✓ BATIMENTS GENERES: {total_buildings} bâtiments")
        print(f"{'='*60}\n")

        self.report({'INFO'}, f"{total_buildings} bâtiment(s) généré(s)")
        return {'FINISHED'}

    def create_building_mesh_vectorial(self, context, contour_2d, roof_height, intensity,
                                       tbvh, props, mr_default, mw_default, name):
        """
        Crée le mesh 3D d'un bâtiment à partir de son contour 2D rectifié.
        Utilise ray_cast sur le terrain pour la base des murs.
        """
        if len(contour_2d) < 3:
            return None

        # Créer le mesh avec BMesh
        bm = bmesh.new()

        # Créer les vertices du toit
        roof_verts = []
        for x, y in contour_2d:
            v = bm.verts.new((x, y, roof_height))
            roof_verts.append(v)

        # Créer la face du toit
        try:
            roof_face = bm.faces.new(roof_verts)
            roof_face.material_index = 0  # Toit
            bmesh.ops.recalc_face_normals(bm, faces=[roof_face])
        except:
            bm.free()
            return None

        # Extrusion vers le bas pour les murs
        edges = list(roof_face.edges)
        ret = bmesh.ops.extrude_edge_only(bm, edges=edges)
        geom = ret['geom']
        verts_ext = [e for e in geom if isinstance(e, bmesh.types.BMVert)]
        faces_ext = [e for e in geom if isinstance(e, bmesh.types.BMFace)]

        # Assigner matériau murs
        for f in faces_ext:
            f.material_index = 1  # Murs

        # Descendre les vertices extrudés au terrain (ray_cast)
        for v in verts_ext:
            # Ray cast depuis au-dessus du toit vers le bas
            origin = Vector((v.co.x, v.co.y, roof_height + 10))
            direction = Vector((0, 0, -1))

            loc, normal, index, dist = tbvh.ray_cast(origin, direction)

            if loc:
                # Projeter sur le terrain
                v.co = loc
            else:
                # Fallback: descendre d'une hauteur par défaut
                v.co.z = roof_height - 3.0

        # Créer le mesh Blender
        mesh = bpy.data.meshes.new(f"Bld_{name}")
        bm.to_mesh(mesh)
        bm.free()

        obj = bpy.data.objects.new(f"Building_{name}", mesh)

        # Appliquer les matériaux procéduraux
        if props.build_apply_materials:
            is_dark_roof = intensity < props.intensity_threshold_roof
            roof_mat_name = f"Roof_{'Tuiles' if is_dark_roof else 'BacAcier'}_{int(intensity*100)}"
            roof_mat = create_roof_material(roof_mat_name, intensity, is_dark_roof)

            gray_factor = random.uniform(0.0, 1.0)
            wall_mat_name = f"Wall_Gray_{int(gray_factor*100)}"
            wall_mat = create_wall_material(wall_mat_name, gray_factor)

            obj.data.materials.append(roof_mat)
            obj.data.materials.append(wall_mat)
        else:
            obj.data.materials.append(mr_default)
            obj.data.materials.append(mw_default)

        return obj

# ==============================================================================
# UI
# ==============================================================================
class LIDARPRO_UL_ObjectList(bpy.types.UIList):
    """Liste des objets LiDAR"""
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if item.obj:
            row = layout.row(align=True)
            row.label(text=item.obj.name, icon='POINTCLOUD_DATA')
            row.prop(item.obj, "hide_viewport", text="",
                    icon='HIDE_OFF' if not item.obj.hide_viewport else 'HIDE_ON',
                    emboss=False)
        else:
            layout.label(text="⚠ Objet Manquant", icon='ERROR')

class LIDARPRO_PT_MainPanel(bpy.types.Panel):
    bl_label = "Lidar Pro V17"
    bl_idname = "LIDARPRO_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Lidar Pro"

    def draw(self, context):
        layout = self.layout
        props = context.scene.lidar_pro

        # SECTION 1: SOURCES DE DONNEES
        box = layout.box()
        row = box.row()
        row.label(text="📦 Sources de Données", icon='IMPORT')

        box.operator("lidarpro.open_ign_map", icon='URL')

        if not check_laspy_installed():
            alert = box.box()
            alert.alert = True
            alert.label(text="⚠ Laspy non installé", icon='ERROR')
            alert.operator("lidarpro.install_laspy", icon='CONSOLE')
        else:
            box.label(text="Filtres d'Import:", icon='FILTER')
            col = box.column(align=True)
            col.prop(props, "import_class_ground", toggle=True)
            col.prop(props, "import_class_veg", toggle=True)
            col.prop(props, "import_class_build", toggle=True)

            box.separator()
            box.prop(props, "import_decimation", slider=True)

            box.separator()
            box.operator("lidarpro.import_las", icon='FILEBROWSER', text="📁 Importer Fichiers LAZ")

            if props.has_master_offset:
                info_box = box.box()
                info_box.label(text=f"📍 Offset: {props.master_offset_x:.2f}, {props.master_offset_y:.2f}", icon='INFO')
                info_box.operator("lidarpro.reset_offset", icon='LOOP_BACK', text="Réinitialiser")

        # SECTION 2: LISTE DES OBJETS
        box = layout.box()
        box.label(text="📋 Objets à Traiter", icon='OUTLINER')

        row = box.row()
        row.template_list("LIDARPRO_UL_ObjectList", "", props, "lidar_list",
                         props, "active_lidar_index", rows=4)

        col = row.column(align=True)
        col.operator("lidarpro.add_to_list", icon='ADD', text="")
        col.operator("lidarpro.remove_from_list", icon='REMOVE', text="")
        col.separator()
        col.operator("lidarpro.toggle_visibility", icon='RESTRICT_VIEW_OFF', text="")

        row = box.row(align=True)
        row.operator("lidarpro.clear_list", icon='TRASH', text="Vider Liste")

        if props.lidar_list:
            stats_box = box.box()
            stats_box.label(text=f"📊 {len(props.lidar_list)} objet(s) dans la liste", icon='INFO')

        layout.separator()

        # SECTION 3: GENERATEURS
        layout.label(text="🔨 Générateurs", icon='GEOMETRY_NODES')

        # TERRAIN
        box = layout.box()
        row = box.row()
        row.label(text="🏔 Terrain", icon='MESH_GRID')

        col = box.column(align=True)
        col.prop(props, "terrain_precision", slider=True)
        col.prop(props, "merge_threshold", slider=True)

        box.separator()
        box.operator("lidarpro.generate_terrain", icon='PLAY', text="▶ Générer Terrain")

        # VEGETATION
        box = layout.box()
        row = box.row()
        row.label(text="🌲 Végétation", icon='OUTLINER_OB_FORCE_FIELD')

        col = box.column(align=True)
        col.prop(props, "veg_density_factor", slider=True)
        col.prop(props, "veg_grid_size")
        col.prop(props, "min_points_veg")

        box.separator()
        box.prop(props, "veg_use_geonodes", icon='GEOMETRY_NODES')

        box.separator()
        box.operator("lidarpro.generate_vegetation", icon='PLAY', text="▶ Générer Végétation")

        # BATIMENTS (VECTORIEL GEOMETRIQUE)
        box = layout.box()
        row = box.row()
        row.label(text="🏠 Bâtiments (Vectoriel)", icon='HOME')

        col = box.column(align=True)
        col.prop(props, "build_alpha_shape", slider=True)
        col.prop(props, "build_dbscan_eps", slider=True)
        col.prop(props, "min_build_points")
        col.prop(props, "min_build_area")

        box.separator()

        # Limite de test
        limit_box = box.box()
        limit_box.prop(props, "build_limit_enabled", icon='EXPERIMENTAL')
        if props.build_limit_enabled:
            limit_box.prop(props, "build_limit_count", slider=True)

        box.separator()

        mat_box = box.box()
        mat_box.prop(props, "build_apply_materials", icon='MATERIAL')
        if props.build_apply_materials:
            mat_box.prop(props, "intensity_threshold_roof", slider=True, text="Seuil Tuiles/Acier")

        box.separator()
        box.operator("lidarpro.generate_buildings", icon='PLAY', text="▶ Générer Bâtiments")

# PANNEAU PARAMETRES AVANCES
class LIDARPRO_PT_AdvancedPanel(bpy.types.Panel):
    bl_label = "Paramètres Avancés"
    bl_idname = "LIDARPRO_PT_advanced"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Lidar Pro"
    bl_parent_id = "LIDARPRO_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        props = context.scene.lidar_pro

        # Terrain avancé
        box = layout.box()
        box.label(text="Terrain", icon='MESH_GRID')
        box.prop(props, "chunk_size")
        box.prop(props, "texture_resolution")
        box.prop(props, "apply_material")

        box.separator()
        gap_box = box.box()
        gap_box.label(text="Comblement des Trous", icon='MESH_DATA')
        gap_box.prop(props, "terrain_fill_gaps")
        if props.terrain_fill_gaps:
            gap_box.prop(props, "terrain_overlap", slider=True)

        # Végétation avancée
        box = layout.box()
        box.label(text="Végétation", icon='OUTLINER_OB_FORCE_FIELD')
        box.prop(props, "height_bush_max")
        box.prop(props, "analyze_radius")
        box.prop(props, "density_threshold")
        box.prop(props, "scale_variation")
        box.prop(props, "rotation_random")

# ==============================================================================
# ENREGISTREMENT
# ==============================================================================
classes = (
    LidarItem,
    LidarProProperties,
    LIDARPRO_UL_ObjectList,
    LIDARPRO_OT_AddToList,
    LIDARPRO_OT_RemoveFromList,
    LIDARPRO_OT_ClearList,
    LIDARPRO_OT_ToggleVisibility,
    LIDARPRO_OT_InstallLaspy,
    LIDARPRO_OT_ResetOffset,
    LIDARPRO_OT_ImportLas,
    LIDARPRO_OT_GenerateTerrain,
    LIDARPRO_OT_GenerateVegetation,
    LIDARPRO_OT_GenerateBuildings,
    LIDARPRO_PT_MainPanel,
    LIDARPRO_PT_AdvancedPanel,
    LIDARPRO_OT_OpenIGNMap
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.lidar_pro = bpy.props.PointerProperty(type=LidarProProperties)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.lidar_pro

if __name__ == "__main__":
    register()
